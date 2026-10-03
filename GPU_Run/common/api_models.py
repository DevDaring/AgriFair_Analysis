"""Frozen frontier panel: hosted models called through a strict per-model fallback chain.

These models are evaluation subjects, never repair targets. GRAFT needs gradients, weights
and hidden states, so a hosted model cannot be localized, adapted or verified with three of
the four readouts. What it can do is answer the questions that need only behaviour: which
direction it fails in, how much of its score a state-blind prior explains, whether its
agronomic advice moves with the farmer's identity, and whether its answer survives a
rotation of the option order.

Each model carries its own route order. Every route is attempted AT MOST ONCE per request;
any failure falls straight through to the next one. There are no retries anywhere, at any
layer, including inside botocore.

    Grok 4.3              xAI direct        ->  OpenRouter key 1  ->  OpenRouter key 2
    GPT-5.6 Luna          OpenAI direct     ->  OpenRouter key 1  ->  OpenRouter key 2
    Nova 2 Lite           Bedrock acct A/B  ->  OpenRouter key 1  ->  OpenRouter key 2
    Nemotron Nano 3 30B   Bedrock acct A/B

The two Bedrock accounts round-robin: they swap starting position on every request, so load
is shared rather than always landing on the first, and a failure on one falls through to the
other before leaving AWS.

Why the two vendor models never touch Bedrock. Verified over two full subscription cycles on
2026-09-04, both AWS accounts return `INVALID_PAYMENT_INSTRUMENT` for every third-party
Marketplace model, tested on Claude Sonnet 5 and GPT-5.6 Luna, before and after making a card
the default payment method. Both accounts are AWS India (AISPL), and the Reserve Bank of
India's restriction on stored card data means AWS Marketplace cannot charge a stored card for
AISPL customers. Amazon's own Nova and the NVIDIA model are unaffected because they bill
through ordinary AWS billing and need no subscription. Routing a vendor model through Bedrock
would therefore burn two guaranteed failures and their latency on every single request, so
those models start at their vendor's own API instead.

Grok 4.3 is deliberately not the newest Grok. xAI currently serves 4.6 and 4.5 at $2.00 and
$6.00 per million tokens, and 4.3 at $1.25 and $2.50, one generation behind at roughly half
the output price. Measured on the real prompts it answers a multiple-choice item in eight
output tokens, so nothing is lost to reasoning overhead, and it replaces Claude Sonnet 5 at
about a third of the cost.

Native call shapes, established by probing the live APIs on 2026-09-05:
  * OpenAI gpt-5.6-luna rejects `max_tokens` and `temperature`; it takes
    `max_completion_tokens` and nothing else. It is a reasoning model, so its output budget
    has to cover the thinking it does before the first visible token.
  * xAI grok-4.3 is OpenAI-compatible and accepts `max_tokens` with `temperature` 0, which
    keeps its decoding consistent with the rest of the study.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import requests

from GPU_Run.common import env_loader
from GPU_Run.common.logging_utils import get_logger

logger = get_logger("api_models")

BEDROCK_REGION = env_loader.get("AWS_REGION", "us-east-1")
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("API_REQUEST_TIMEOUT_SECONDS", "180"))
OPENAI_BASE_URL = os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_API_BASE_URL", "https://api.anthropic.com/v1")
ANTHROPIC_VERSION = "2023-06-01"

# Route family names used in ApiModelSpec.route_order.
BEDROCK, ANTHROPIC, OPENAI, XAI, OPENROUTER = "bedrock", "anthropic", "openai", "xai", "openrouter"
XAI_BASE_URL = os.environ.get("XAI_API_BASE_URL", "https://api.x.ai/v1")


@dataclass(frozen=True)
class ApiModelSpec:
    """One hosted model, the ids it is known by on each provider, and its route order."""
    tier: str                       # results tier name; carries no underscore, since the
                                    # prediction-filename parser splits on them
    display_name: str
    vendor: str
    route_order: Tuple[str, ...]
    bedrock_model_id: Optional[str] = None
    openrouter_model_id: Optional[str] = None
    native_model_id: Optional[str] = None
    # Output budgets. A reasoning model spends most of its allowance before it emits a
    # visible token, so a budget tuned for a plain model silently truncates it to an empty
    # string and the study records a parse failure that is really a truncation.
    answer_max_tokens: int = 0      # 0 means use the study default
    advice_max_tokens: int = 0
    notes: str = ""


REGISTRY: Dict[str, ApiModelSpec] = {
    "frontier-grok-4-3": ApiModelSpec(
        tier="frontier-grok-4-3",
        display_name="Grok 4.3",
        vendor="xai",
        route_order=(XAI, OPENROUTER),
        native_model_id="grok-4.3",
        openrouter_model_id="x-ai/grok-4.3",
        notes="xAI direct is primary. One generation behind 4.5 and 4.6 and priced at $1.25 "
              "and $2.50 per million against their $2.00 and $6.00. Answers a multiple-choice "
              "item in eight output tokens, so the saving is not paid for in reasoning overhead.",
    ),
    "frontier-gpt-5-6-luna": ApiModelSpec(
        tier="frontier-gpt-5-6-luna",
        display_name="GPT-5.6 Luna",
        vendor="openai",
        route_order=(OPENAI, OPENROUTER),
        native_model_id="gpt-5.6-luna",
        openrouter_model_id="openai/gpt-5.6-luna",
        bedrock_model_id="us.openai.gpt-5.6-luna",       # recorded, deliberately not routed
        answer_max_tokens=4096,
        advice_max_tokens=4096,
        notes="OpenAI direct is primary. Takes max_completion_tokens only, no temperature. "
              "Measured over 1000 reasoning tokens before its first visible token, so its "
              "budgets are several times the others'.",
    ),
    "frontier-nova-2-lite": ApiModelSpec(
        tier="frontier-nova-2-lite",
        display_name="Nova 2 Lite",
        vendor="amazon",
        route_order=(BEDROCK, OPENROUTER),
        bedrock_model_id="us.amazon.nova-2-lite-v1:0",
        openrouter_model_id="amazon/nova-2-lite-v1",
        notes="Amazon's own model, so it bills through ordinary AWS billing and needs no "
              "Marketplace subscription. Inference profile only, hence the us. prefix. Wraps "
              "JSON in code fences, which the deterministic parser already strips.",
    ),
    "frontier-kimi-k2-5": ApiModelSpec(
        tier="frontier-kimi-k2-5",
        display_name="Kimi K2.5",
        vendor="moonshot",
        route_order=(BEDROCK, OPENROUTER),
        bedrock_model_id="moonshotai.kimi-k2.5",
        openrouter_model_id="moonshotai/kimi-k2.5",
        notes="On-demand on Bedrock, both accounts. Probed 2026-09-21: Hindi/Bengali MCQ answers in "
              "10 output tokens, advice in the target script (91% / 100%), 627 req/min sustained at "
              "concurrency 20 with no throttling. $0.60 / $3.00 per M tokens.",
    ),
    "frontier-glm-4-7": ApiModelSpec(
        tier="frontier-glm-4-7",
        display_name="GLM-4.7",
        vendor="zai",
        route_order=(BEDROCK, OPENROUTER),
        bedrock_model_id="zai.glm-4.7",
        openrouter_model_id="z-ai/glm-4.7",
        notes="On-demand on Bedrock, both accounts. Probed 2026-09-21: wraps the JSON in code "
              "fences (parser strips them), Hindi 85% / Bengali 100% target script, 2,100 req/min "
              "sustained. $0.60 / $2.20 per M tokens.",
    ),
    "frontier-qwen3-next-80b": ApiModelSpec(
        tier="frontier-qwen3-next-80b",
        display_name="Qwen3-Next 80B A3B",
        vendor="qwen",
        route_order=(BEDROCK, OPENROUTER),
        bedrock_model_id="qwen.qwen3-next-80b-a3b",
        openrouter_model_id="qwen/qwen3-next-80b-a3b-instruct",
        notes="Registered as the cheap alternative (not in the default panel): $0.15 / $1.20, "
              "2,900 req/min, 9-token MCQ answers, Hindi 89% / Bengali 100%.",
    ),
    "frontier-nemotron-nano-3-30b": ApiModelSpec(
        tier="frontier-nemotron-nano-3-30b",
        display_name="Nemotron Nano 3 30B",
        vendor="nvidia",
        route_order=(BEDROCK,),
        bedrock_model_id="nvidia.nemotron-nano-3-30b",
        notes="On-demand on Bedrock, both accounts. Absent from the OpenRouter catalogue and "
              "with no native route configured, so the two Bedrock accounts are all it has.",
    ),
}


def active_api_models() -> List[str]:
    """Tiers to evaluate. API_SUBJECT_MODELS restricts the set."""
    override = os.environ.get("API_SUBJECT_MODELS")
    if not override:
        return list(REGISTRY)
    wanted = [t.strip() for t in override.split(",") if t.strip()]
    unknown = [t for t in wanted if t not in REGISTRY]
    if unknown:
        raise SystemExit(f"Unknown API subject model(s): {unknown}. Known: {sorted(REGISTRY)}")
    return wanted


def answer_token_budget(tier: str, default: int) -> int:
    return REGISTRY[tier].answer_max_tokens or default


def advice_token_budget(tier: str, default: int) -> int:
    return REGISTRY[tier].advice_max_tokens or default


def route_plan() -> Dict[str, List[str]]:
    """The configured route order per model, for logging and for the run metadata."""
    return {tier: list(spec.route_order) for tier, spec in REGISTRY.items()}


# ------------------------------- the chain ----------------------------------

@dataclass
class RouteAttempt:
    route: str
    ok: bool
    error: str = ""


@dataclass
class ApiResponse:
    text: str
    route: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    attempts: List[RouteAttempt] = field(default_factory=list)

    @property
    def failed_routes(self) -> str:
        return ";".join(a.route for a in self.attempts if not a.ok)


class ApiRouter:
    """Per-model route chains, one attempt per route, no retries anywhere."""

    def __init__(self):
        self._lock = threading.Lock()
        self._round_robin_index = 0
        self._bedrock_clients: Dict[str, object] = {}
        self.route_counts: Dict[str, int] = {}
        self.failure_counts: Dict[str, int] = {}
        self.aws_accounts = self._aws_accounts()
        self.openrouter_keys = self._openrouter_keys()
        self.openrouter_base = env_loader.get("OPENROUTER_API_BASE_URL", "https://openrouter.ai/api/v1")
        self.openai_key = env_loader.get("OPENAI_DIRECT_KEY")
        self.anthropic_key = env_loader.get("ANTHROPIC_DIRECT_KEY")
        self.anthropic_workspace = env_loader.get("ANTHROPIC_WORKSPACE_ID")
        self.xai_key = env_loader.get("XAI_DIRECT_KEY")

    # ---- credentials, read only through env_loader and never logged ----

    @staticmethod
    def _aws_accounts() -> List[Tuple[str, str, str]]:
        pairs = [("aws-account-1", env_loader.get("AWS_ACCESS_KEY_1"), env_loader.get("AWS_SECRET_KEY_1")),
                 ("aws-account-2", env_loader.get("AWS_ACCESS_KEY_2"), env_loader.get("AWS_SECRET_KEY_2"))]
        return [(n, a, s) for n, a, s in pairs if a and s]

    @staticmethod
    def _openrouter_keys() -> List[Tuple[str, str]]:
        pairs = [("openrouter-key-1", env_loader.get("OPENROUTER_KEY1")),
                 ("openrouter-key-2", env_loader.get("OPENROUTER_KEY2"))]
        return [(n, k) for n, k in pairs if k]

    def available(self) -> bool:
        return bool(self.aws_accounts or self.openrouter_keys or self.openai_key
                    or self.anthropic_key or self.xai_key)

    def describe_routes(self, tier: str) -> List[str]:
        """The routes this model would actually try, in order, given the credentials present."""
        return [name for name, _ in self._ordered_routes(REGISTRY[tier], advance=False)]

    def _bedrock_client(self, account_name: str, access_key: str, secret_key: str):
        if account_name not in self._bedrock_clients:
            import boto3
            from botocore.config import Config

            # max_attempts=1 is what makes "no retries" true at the transport layer as well
            # as in the loop below; without it botocore silently retries throttles.
            self._bedrock_clients[account_name] = boto3.client(
                "bedrock-runtime",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=BEDROCK_REGION,
                config=Config(retries={"max_attempts": 1, "mode": "standard"},
                              connect_timeout=15, read_timeout=REQUEST_TIMEOUT_SECONDS),
            )
        return self._bedrock_clients[account_name]

    # ---- the four transports ----

    def _call_bedrock(self, account, spec: ApiModelSpec, prompt: str, max_tokens: int):
        name, access_key, secret_key = account
        client = self._bedrock_client(name, access_key, secret_key)
        response = client.converse(
            modelId=spec.bedrock_model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
        )
        text = "".join(block.get("text", "") for block in response["output"]["message"]["content"])
        usage = response.get("usage", {})
        return text, int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0))

    def _call_openai(self, spec: ApiModelSpec, prompt: str, max_tokens: int):
        """OpenAI Chat Completions. gpt-5.6 takes max_completion_tokens and rejects
        temperature, so neither max_tokens nor temperature is sent."""
        response = requests.post(
            f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"},
            json={"model": spec.native_model_id,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_completion_tokens": max_tokens},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        if "choices" not in payload:
            raise RuntimeError(str(payload.get("error", payload))[:200])
        text = payload["choices"][0]["message"].get("content") or ""
        usage = payload.get("usage", {})
        return text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))

    def _call_anthropic(self, spec: ApiModelSpec, prompt: str, max_tokens: int):
        """Anthropic Messages. The key is workspace-scoped, and temperature is deprecated
        for this model, so it is omitted rather than set to zero."""
        headers = {"x-api-key": self.anthropic_key, "anthropic-version": ANTHROPIC_VERSION,
                   "content-type": "application/json"}
        if self.anthropic_workspace:
            headers["anthropic-workspace-id"] = self.anthropic_workspace
        response = requests.post(
            f"{ANTHROPIC_BASE_URL.rstrip('/')}/messages",
            headers=headers,
            json={"model": spec.native_model_id, "max_tokens": max_tokens,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        text = "".join(block.get("text", "") for block in payload.get("content", []))
        usage = payload.get("usage", {})
        return text, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))

    def _call_xai(self, spec: ApiModelSpec, prompt: str, max_tokens: int):
        """xAI is OpenAI-compatible and accepts temperature, so decoding stays at zero like
        every other arm in the study."""
        response = requests.post(
            f"{XAI_BASE_URL.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.xai_key}", "Content-Type": "application/json"},
            json={"model": spec.native_model_id,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0, "max_tokens": max_tokens},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        if "choices" not in payload:
            raise RuntimeError(str(payload.get("error", payload))[:200])
        text = payload["choices"][0]["message"].get("content") or ""
        usage = payload.get("usage", {})
        return text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))

    def _call_openrouter(self, key: str, spec: ApiModelSpec, prompt: str, max_tokens: int):
        response = requests.post(
            f"{self.openrouter_base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": spec.openrouter_model_id,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0, "max_tokens": max_tokens},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        if "choices" not in payload:
            raise RuntimeError(str(payload.get("error", payload))[:200])
        text = payload["choices"][0]["message"].get("content") or ""
        usage = payload.get("usage", {})
        return text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))

    # ---- route assembly ----

    def _ordered_routes(self, spec: ApiModelSpec, advance: bool = True):
        """(route_name, callable(prompt, max_tokens)) in the model's configured order.

        A family whose credentials or model id are missing contributes nothing, so a
        half-configured environment degrades to the routes that can actually work."""
        routes: List[Tuple[str, Callable[[str, int], Tuple[str, int, int]]]] = []
        for family in spec.route_order:
            if family == BEDROCK and spec.bedrock_model_id and self.aws_accounts:
                accounts = self.aws_accounts
                if len(accounts) > 1:
                    if advance:
                        with self._lock:
                            start = self._round_robin_index % len(accounts)
                            self._round_robin_index += 1
                    else:
                        start = self._round_robin_index % len(accounts)
                    accounts = accounts[start:] + accounts[:start]
                for account in accounts:
                    routes.append((account[0],
                                   lambda p, m, a=account: self._call_bedrock(a, spec, p, m)))
            elif family == ANTHROPIC and spec.native_model_id and self.anthropic_key:
                routes.append(("anthropic-direct", lambda p, m: self._call_anthropic(spec, p, m)))
            elif family == OPENAI and spec.native_model_id and self.openai_key:
                routes.append(("openai-direct", lambda p, m: self._call_openai(spec, p, m)))
            elif family == XAI and spec.native_model_id and self.xai_key:
                routes.append(("xai-direct", lambda p, m: self._call_xai(spec, p, m)))
            elif family == OPENROUTER and spec.openrouter_model_id:
                for name, key in self.openrouter_keys:
                    routes.append((name, lambda p, m, k=key: self._call_openrouter(k, spec, p, m)))
        return routes

    def complete(self, tier: str, prompt: str, max_tokens: int = 64) -> Optional[ApiResponse]:
        """One prompt through the chain. Returns None only when every route has failed."""
        spec = REGISTRY[tier]
        attempts: List[RouteAttempt] = []
        started = time.time()
        for route_name, call in self._ordered_routes(spec):
            try:
                text, in_tokens, out_tokens = call(prompt, max_tokens)
                attempts.append(RouteAttempt(route_name, True))
                with self._lock:
                    self.route_counts[route_name] = self.route_counts.get(route_name, 0) + 1
                return ApiResponse(text=text, route=route_name, input_tokens=in_tokens,
                                   output_tokens=out_tokens, latency_seconds=time.time() - started,
                                   attempts=attempts)
            except Exception as e:  # no retry: record and fall straight through
                detail = _error_detail(e)
                attempts.append(RouteAttempt(route_name, False, detail))
                with self._lock:
                    self.failure_counts[route_name] = self.failure_counts.get(route_name, 0) + 1
                logger.warning("%s via %s failed (%s); falling through to the next route.",
                               spec.display_name, route_name, detail)
        logger.error("%s: every route failed (%s).", spec.display_name,
                     ", ".join(f"{a.route}={a.error}" for a in attempts))
        return None

    def usage_summary(self) -> Dict[str, object]:
        return {"requests_served_by_route": dict(self.route_counts),
                "failures_by_route": dict(self.failure_counts),
                "configured_route_order": route_plan()}


def _error_detail(exc: Exception) -> str:
    """A short, safe error label. Never includes a credential or a full response body."""
    response = getattr(exc, "response", None)
    if isinstance(response, dict) and "Error" in response:
        return str(response["Error"].get("Code", type(exc).__name__))
    status = getattr(response, "status_code", None)
    if status is not None:
        return f"HTTP{status}"
    return type(exc).__name__
