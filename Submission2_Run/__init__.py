"""Submission 2: three-pillar audit of commercial LLM advisory services (API only, no training).

Run from Codes/:

    python -m Submission2_Run.build_kcc_sample --csv <kcc_export.csv>      # 4.1 real queries
    python -m Submission2_Run.build_facts --values <archived_table.csv>    # 4.3 documented differences
    python -m Submission2_Run.build_pairs                                  # 4.2 identity toggles
    python -m Submission2_Run.run_experiments --smoke                      # 5 items per model, offline fake router
    python -m Submission2_Run.run_experiments --stage all                  # needs api_enabled: true in config.yaml
    python -m Submission2_Run.analysis
    python -m Submission2_Run.human_study --prepare
    python -m Submission2_Run.report

Everything the models are sent goes through Submission2_Run.providers.Router, which wraps the
frozen route chain in GPU_Run/common/api_models.py (xAI / OpenAI / Bedrock -> OpenRouter key 1 ->
OpenRouter key 2). Every response is cached by prompt hash; no item is ever re-queried.
Metric definitions, the error taxonomy and the cluster statistics are imported from Next_Run.
"""
