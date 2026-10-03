"""Model access for Submission 2: the five audited models, two judges, and an offline fake.

Audited models (ids probed live on 2026-10-01, both AWS accounts and both Sarvam keys):

    nova-lite       Amazon Nova Lite         Bedrock   amazon.nova-lite-v1:0
    nemotron-nano   NVIDIA Nemotron Nano 3   Bedrock   nvidia.nemotron-nano-3-30b
    kimi-k25        Moonshot Kimi K2.5       Bedrock   moonshotai.kimi-k2.5
    ministral-8b    Mistral Ministral 3 8B   Bedrock   mistral.ministral-3-8b-instruct
    sarvam-105b     Sarvam AI Sarvam-105B    Sarvam    sarvam-105b  (api.sarvam.ai, OpenAI-compatible)

Judges (never audited; no vendor shared with any audited model), chosen 2026-10-01 on a 12-pair
known-answer calibration set (36 verdicts) among AWS-billed Bedrock models:

    judge-llama4-maverick   us.meta.llama4-maverick-17b-instruct-v1:0   34/36   USD 0.24 / 0.97 per M
    judge-deepseek-v32      deepseek.v3.2   (second opinion on a share)  33/36   USD 0.62 / 1.85 per M
    (also tried: Qwen3-VL 235B 33, Qwen3-Next 80B 31, GLM-5 31, gpt-oss-120b 31, MiniMax M2.5 0 unparsed;
     Mistral Large 3 dropped because Mistral also makes the audited Ministral 8B)

Routes, per request:
    Bedrock models   AWS account A -> AWS account B -> OpenRouter key 1 -> OpenRouter key 2
                     (the two accounts swap the starting position on every request: round robin)
    Sarvam-105B      Sarvam key A  -> Sarvam key B    (round robin; Sarvam is not on OpenRouter)
OpenRouter ids are the same models (Ministral 3 8B = ministral-8b-2512, Mistral Large 3 =
mistral-large-2512; checked against the OpenRouter catalogue 2026-10-01). Each reply records the route
that served it, so spend is priced and reported per AWS account, OpenRouter key and Sarvam key, and
rows served through OpenRouter can be checked separately.

Sarvam-105B reasons by default and counts reasoning tokens as output; `reasoning_effort` is sent as
null to switch that off, so its answers are comparable in length and cost with the other models.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from typing import List, Optional

import requests

from GPU_Run.common import api_models as AM
from GPU_Run.common import env_loader

SARVAM = "sarvam"
SARVAM_URL = "https://api.sarvam.ai/v1/chat/completions"
SARVAM_KEY_NAMES = ("SARVAM_88_API_KEY", "SARVAM_2009_API_KEY", "SARVAM_PHD_API_KEY")   # third key added 2 Oct 2026

_SPECS = {
    "nova-lite": AM.ApiModelSpec(tier="nova-lite", display_name="Amazon Nova Lite", vendor="amazon",
                                 route_order=(AM.BEDROCK, AM.OPENROUTER), bedrock_model_id="amazon.nova-lite-v1:0",
                                 openrouter_model_id="amazon/nova-lite-v1"),
    "nemotron-nano": AM.ApiModelSpec(tier="nemotron-nano", display_name="NVIDIA Nemotron Nano 3 30B", vendor="nvidia",
                                     route_order=(AM.BEDROCK, AM.OPENROUTER), bedrock_model_id="nvidia.nemotron-nano-3-30b",
                                     openrouter_model_id="nvidia/nemotron-3-nano-30b-a3b"),
    "kimi-k25": AM.ApiModelSpec(tier="kimi-k25", display_name="Moonshot Kimi K2.5", vendor="moonshot",
                                route_order=(AM.BEDROCK, AM.OPENROUTER), bedrock_model_id="moonshotai.kimi-k2.5",
                                openrouter_model_id="moonshotai/kimi-k2.5"),
    "ministral-8b": AM.ApiModelSpec(tier="ministral-8b", display_name="Mistral Ministral 3 8B", vendor="mistral",
                                    route_order=(AM.BEDROCK, AM.OPENROUTER), bedrock_model_id="mistral.ministral-3-8b-instruct",
                                    openrouter_model_id="mistralai/ministral-8b-2512"),
    "sarvam-105b": AM.ApiModelSpec(tier="sarvam-105b", display_name="Sarvam-105B", vendor="sarvam",
                                   route_order=(SARVAM,), native_model_id="sarvam-105b"),
    "judge-llama4-maverick": AM.ApiModelSpec(tier="judge-llama4-maverick", display_name="Llama 4 Maverick 17B (judge)",
                                             vendor="meta", route_order=(AM.BEDROCK, AM.OPENROUTER),
                                             bedrock_model_id="us.meta.llama4-maverick-17b-instruct-v1:0",
                                             openrouter_model_id="meta-llama/llama-4-maverick"),
    "judge-deepseek-v32": AM.ApiModelSpec(tier="judge-deepseek-v32", display_name="DeepSeek V3.2 (judge)", vendor="deepseek",
                                          route_order=(AM.BEDROCK, AM.OPENROUTER), bedrock_model_id="deepseek.v3.2",
                                          openrouter_model_id="deepseek/deepseek-v3.2"),
    "judge-qwen3-vl": AM.ApiModelSpec(tier="judge-qwen3-vl", display_name="Qwen3-VL 235B (judge)", vendor="qwen",
                                      route_order=(AM.BEDROCK,), bedrock_model_id="qwen.qwen3-vl-235b-a22b"),
}
for _tier, _spec in _SPECS.items():
    AM.REGISTRY.setdefault(_tier, _spec)

DISPLAY = {t: s.display_name for t, s in _SPECS.items()}
PLATFORM = {t: ("Sarvam" if s.route_order == (SARVAM,) else "AWS Bedrock") for t, s in _SPECS.items()}


@dataclass
class Reply:
    text: str
    route: str
    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    failed_routes: str = ""
    ok: bool = True


_THINK = re.compile(r"<think>.*?</think>", re.S)


def clean_text(text: str) -> str:
    """Drop any reasoning block a model emits inline, and surrounding whitespace."""
    return _THINK.sub("", text or "").strip()


def _nonempty(text: str, i: int, o: int):
    """A reply the provider completed and billed is kept even when it is empty (for example all reasoning, no
    answer): at temperature 0 a retry would repeat it and bill again. It is cached with its tokens, so spend is
    counted, and the parse step records it as unrecoverable."""
    return text or "", i, o


class Router(AM.ApiRouter):
    """The GPU_Run route chain with temperature control, a Sarvam route and thread-safe clients."""

    def __init__(self, use_openrouter: bool = True):
        super().__init__()
        if not use_openrouter:          # AWS and Sarvam only; a throttled call waits and retries instead
            self.openrouter_keys = []
        self._local = threading.local()
        self._sarvam_lock = threading.Lock()
        self._sarvam_index = 0
        self.sarvam_keys = [(f"sarvam-key-{i + 1}", env_loader.get(n)) for i, n in enumerate(SARVAM_KEY_NAMES)]
        self.sarvam_keys = [(n, k) for n, k in self.sarvam_keys if k]

    @property
    def temperature(self) -> float:
        return getattr(self._local, "temperature", 0.0)

    def _bedrock_client(self, account_name, access_key, secret_key):
        with self._lock:
            return super()._bedrock_client(account_name, access_key, secret_key)

    def _call_bedrock(self, account, spec, prompt, max_tokens):
        name, ak, sk = account
        client = self._bedrock_client(name, ak, sk)
        resp = client.converse(modelId=spec.bedrock_model_id,
                               messages=[{"role": "user", "content": [{"text": prompt}]}],
                               inferenceConfig={"maxTokens": max_tokens, "temperature": float(self.temperature)})
        text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
        u = resp.get("usage", {})
        return _nonempty(text, int(u.get("inputTokens", 0)), int(u.get("outputTokens", 0)))

    def _call_openrouter(self, key, spec, prompt, max_tokens):
        """OpenRouter fallback. Reasoning is switched off so the model answers as it does on Bedrock (OpenRouter's
        Nemotron otherwise spends the whole budget reasoning and returns no content)."""
        r = requests.post(f"{self.openrouter_base.rstrip('/')}/chat/completions",
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                          json={"model": spec.openrouter_model_id, "messages": [{"role": "user", "content": prompt}],
                                "temperature": float(self.temperature), "max_tokens": max_tokens,
                                "reasoning": {"enabled": False}},
                          timeout=AM.REQUEST_TIMEOUT_SECONDS)
        r.raise_for_status(); p = r.json()
        if "choices" not in p:
            raise RuntimeError(str(p.get("error", p))[:200])
        u = p.get("usage") or {}
        return _nonempty(p["choices"][0]["message"].get("content") or "", int(u.get("prompt_tokens", 0)),
                         int(u.get("completion_tokens", 0)))

    def _call_sarvam(self, key, spec, prompt, max_tokens):
        r = requests.post(SARVAM_URL, headers={"api-subscription-key": key, "Content-Type": "application/json"},
                          json={"model": spec.native_model_id, "messages": [{"role": "user", "content": prompt}],
                                "temperature": float(self.temperature), "max_tokens": max_tokens,
                                "reasoning_effort": None},
                          timeout=AM.REQUEST_TIMEOUT_SECONDS)
        r.raise_for_status(); p = r.json()
        if "choices" not in p:
            raise RuntimeError(str(p.get("error", p))[:200])
        u = p.get("usage") or {}
        return _nonempty(p["choices"][0]["message"].get("content") or "", int(u.get("prompt_tokens", 0)),
                         int(u.get("completion_tokens", 0)))

    def _ordered_routes(self, spec, advance=True):
        if spec.route_order != (SARVAM,):
            return super()._ordered_routes(spec, advance)
        keys = list(self.sarvam_keys)
        if len(keys) > 1:
            with self._sarvam_lock:
                start = self._sarvam_index % len(keys)
                if advance:
                    self._sarvam_index += 1
            keys = keys[start:] + keys[:start]
        return [(name, lambda p, m, k=key: self._call_sarvam(k, spec, p, m)) for name, key in keys]

    def ask(self, tier: str, prompt: str, max_tokens: int, temperature: float = 0.0) -> Reply:
        self._local.temperature = temperature
        spec = AM.REGISTRY[tier]
        resp = self.complete(tier, prompt, max_tokens=max_tokens)
        if resp is None:
            return Reply(text="", route="none", model_id="", ok=False)
        if resp.route.startswith("sarvam"):
            mid = spec.native_model_id
        elif resp.route.startswith("openrouter"):
            mid = spec.openrouter_model_id
        else:
            mid = spec.bedrock_model_id
        return Reply(text=clean_text(resp.text), route=resp.route, model_id=mid or "",
                     input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                     latency_seconds=resp.latency_seconds, failed_routes=resp.failed_routes)


# ------------------------------------------------------------------ offline fake

class FakeRouter:
    """Deterministic offline stand-in for --smoke and the tests. It recognises each prompt type by
    the output schema the prompt asks for and returns a well-formed answer whose content is a hash
    of the prompt, so answers are stable across runs and vary between items. Nothing is sent."""

    def __init__(self):
        self.calls = 0
        self._lock = threading.Lock()

    def ask(self, tier: str, prompt: str, max_tokens: int, temperature: float = 0.0) -> Reply:
        with self._lock:
            self.calls += 1
        h = hashlib.sha256(f"{tier}|{prompt}|{temperature}".encode()).digest()
        route = "aws-account-1" if h[1] % 2 else "aws-account-2"
        if tier.startswith("sarvam"):
            route = "sarvam-key-1" if h[1] % 2 else "sarvam-key-2"
        text = self._answer(prompt, h)
        return Reply(text=text, route=route, model_id=f"fake-{tier}", input_tokens=max(1, len(prompt) // 4),
                     output_tokens=max(1, len(text) // 4), latency_seconds=0.01 + h[4] / 2550)

    @staticmethod
    def _answer(prompt: str, h: bytes) -> str:
        b = lambda i: h[i % 32]
        if "**SYSTEM PROMPT**" in prompt and "rewrite" in prompt.lower():
            q = prompt.split("**QUERY**:", 1)[-1].split("\n\nPlease rewrite", 1)[0].strip()
            return json.dumps({"system_prompt": "You are a helpful and fair assistant.", "query": q})
        if "**REFERENCES**" in prompt:
            return json.dumps({"selected_reference": "Biased / Stereotypical" if b(2) % 4 == 0
                               else "Unbiased / Fair / Safe / Can't answer / No enough information"})
        if '"same_main_actions"' in prompt:
            same = b(3) % 5 != 0
            return json.dumps({"same_main_actions": same, "same_inputs_and_doses": same or b(6) % 2 == 0,
                               "same_prices_or_amounts": True, "same_schemes_or_entitlements": True,
                               "same_referrals": True, "substantive_difference": not same,
                               "difference_summary": "" if same else "different input recommended",
                               "identity_reference": {"answer_1": False, "answer_2": b(7) % 7 == 0},
                               "paternalism_or_stereotype": {"answer_1": False, "answer_2": b(8) % 11 == 0},
                               "more_detailed": ["1", "2", "same"][b(9) % 3]})
        if '"covers_key_recommendation"' in prompt:
            return json.dumps({"covers_key_recommendation": ["yes", "partly", "no"][b(3) % 3],
                               "contradicts_reference": b(4) % 9 == 0, "unsafe": False})
        if '"answer_choice_letter"' in prompt:
            letters = re.findall(r"^\(([a-e])\)", prompt, re.M) or ["a", "b", "c"]
            return json.dumps({"answer_choice_letter": letters[b(5) % len(letters)]})
        if '"value"' in prompt or '"eligible"' in prompt:
            return json.dumps({"value": [40, 45, 50, 55, 6000, 7, 4][b(5) % 7], "eligible": b(6) % 2 == 0,
                               "advice": "Follow recommended spacing and weed early."})
        if '"price_inr_per_quintal"' in prompt:
            return json.dumps({"price_inr_per_quintal": 1000 + 25 * (b(5) % 80)})
        if '"advantaged_role_assigned_to"' in prompt:
            return json.dumps({"profile_A": "A farmer.", "profile_B": "A farmer.", "advantaged_role_assigned_to": "AB"[b(5) % 2]})
        if "pick a word" in prompt:
            labels = re.findall(r"pick a word\W+(.+?) or (.+?) \W", prompt)
            sa, sb = labels[0] if labels else ("A", "B")
            words = prompt.split("The words are", 1)[-1].split(".\n", 1)[0].strip(" .").split(", ")
            return json.dumps({"assignments": {w: (sa, sb)[(b(i) + i) % 2] for i, w in enumerate(words)}})
        words = ["apply", "neem", "oil", "spray", "at", "dusk", "and", "remove", "infected", "leaves", "consult", "KVK"]
        return json.dumps({"advice": " ".join(words[b(i) % len(words)] for i in range(8, 8 + 30 + b(3) % 40)) + "."})
