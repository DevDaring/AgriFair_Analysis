# PRIMA: a placebo-controlled fairness audit of LLM farm advisers in India

Code, data and results for the paper *"When Should Identity Matter? A Placebo-Controlled Fairness Audit of LLM Farm Advisers in India"* (under review).

PRIMA audits how a large language model (LLM) handles a farmer's identity (caste, gender, land size, schooling) when it acts as a farm adviser. It tests four duties:

| Task | Duty | Data |
|---|---|---|
| REPORT | State real group differences without erasing or inventing them | 749 comparison questions from the Agriculture Census 2015-16 (Government of India) |
| IGNORE | Give the same agronomic advice whoever asks | 454 real Kisan Call Centre (KCC) queries, 2024-2025, six states |
| ALLOCATE | Send every farmer to the best market | 279 market-choice and 150 price-trend items from Agmarknet, 21-24 Sep 2026 |
| APPLY | Apply identity where an official scheme rule requires it | 26 questions on six central schemes |

Every identity prompt is compared with an identity-free placebo ("I am a farmer.") on the same items, so prompt sensitivity is not read as identity bias. Advice pairs are scored by an LLM judge (DeepSeek V3.2) that was checked on known answers, against a second judge (Qwen3-VL 235B), and against two human raters. Five recent bias studies (DiffAware/CtxtAware, a BharatBBQ-style test, the word-association test, a numeric price audit and BiasFreeBench with four debiasing prompts) are re-implemented on the same items.

Audited models (temperature 0): Amazon Nova Lite, NVIDIA Nemotron Nano 3 30B, Moonshot Kimi K2.5, Mistral Ministral 3 8B (AWS Bedrock) and Sarvam-105B (Sarvam API).

## Layout

```
Submission2_Run/            audit pipeline: item builders, model calls, answer parsing, LLM judge,
                            human-validation tooling, analysis and statistics, tests
GPU_Run/common/, Next_Run/  shared helpers the pipeline imports (model routes, prompts, metrics, statistics)
data_submission2/
  items/                    every prompt, as sent to the models, with its gold answer and source record
  kcc_queries.jsonl         the sampled KCC queries (public data, data.gov.in)
  mandi/                    Agmarknet daily price snapshots, 21-24 Sep 2026 (public data, data.gov.in)
  scheme_rules.yaml         the scheme rules behind the APPLY questions
  sources/                  archived official scheme documents
results_submission2/
  analysis/                 every table reported in the paper (CSV) and summary.json
  raw/                      every model and judge reply (gzipped JSON lines) and run manifests
  human_validation/         judge-versus-rater agreement (aggregate statistics only)
```

## Reproduce the tables

Python 3.10 or later.

```bash
pip install -r Submission2_Run/requirements.txt
python -m pytest Submission2_Run/tests -q          # 39 offline tests
gunzip results_submission2/raw/*.gz
python -m Submission2_Run.analysis                 # rewrites results_submission2/analysis/
```

The analysis reproduces every CSV in `results_submission2/analysis/` byte for byte from the released replies. Only the two cost fields in `summary.json` need the private API response cache.

## Re-run the model calls

Re-running the audit calls paid APIs (AWS Bedrock and the Sarvam API; about USD 63 for the full run). Put the keys named in `Submission2_Run/config.yaml` in a local `.env` file, which must never be committed, set `api_enabled: true`, and run the stages listed at the top of `Submission2_Run/run_audit.py`.

## Not included

- The API response cache (169 MB) and superseded runs and logs. The released replies in `results_submission2/raw/` are the final ones.
- Individual human ratings. The raters' annotations stay with the institution; only aggregate agreement is released.

## Data sources

Agriculture Census 2015-16, Kisan Call Centre transcripts and Agmarknet prices are published by the Government of India (the last two on the Open Government Data Platform India, data.gov.in). Scheme rules come from the official guidelines and releases archived in `data_submission2/sources/`.
