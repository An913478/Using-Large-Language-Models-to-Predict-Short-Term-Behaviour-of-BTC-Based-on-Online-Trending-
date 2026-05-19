"""Smoke-test runner for the pipeline scripts.

This lightweight utility verifies that each pipeline script can be imported
and (optionally) executed in a controlled "safe" mode. It is intended for
CI pre-checks and developer quick checks before running the full pipeline.

Behavior
--------
- By default the tool only attempts to import each script and confirms a
  `main` call-able exists.
- If `--run` is passed, the tool will call `main()` only for modules that
  declare `SMOKE_TEST_SAFE = True` at module scope. This protects against
  accidentally executing long-running or network-heavy steps.

Usage
-----
Import-check only:

    python scripts/smoke_test_pipeline.py

Attempt to run safe mains:

    python scripts/smoke_test_pipeline.py --run

Add new script entries to `SCRIPTS_TO_CHECK` below.
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


SCRIPTS_TO_CHECK: List[str] = [
    # Data acquisition
    "fetch_market_data",
    "fetch_google_trends",
    "fetch_marketaux_news",
    "aggregate_marketaux_daily",
    # Market-attention merge and feature engineering
    "merge_market_trends",
    "feature_engineering",
    "context_feature_expansion",
    # LLM extraction
    "extract_news_llm_features_openai",
    "extract_news_llm_features_gemini",
    "extract_news_llm_features_claude",
    # Market + LLM merge and augmentation
    "merge_market_llm_features",
    "feature_engineering_phase1_plus",
    # Provider-specific and ensemble features
    "merge_market_llm_features_openai",
    "merge_market_llm_features_gemini",
    "merge_market_llm_features_claude",
    "merge_llm_ensemble_features",
    "merge_llm_uncertainty_features",
    "merge_llm_weighted_ensemble",
    # Modelling
    "feature_engineering_multi_horizon",
    "train_multi_horizon_baselines",
    "train_multi_horizon_transformer",
    "train_attention_context_model",
    "train_return_direction_dcn",
    "train_multi_horizon_llm_compare",
    "train_multi_horizon_ensemble",
    "train_final_best_model",
    # Evaluation & utilities
    "aggregate_repeated_seed_results",
    "benchmark_skill_heatmap_validation",
    "evaluate_embargo_sensitivity",
    "evaluate_trading_strategy",
    "evaluate_uncertainty_trading_strategy",
]


@dataclass
class CheckResult:
    name: str
    imported: bool = False
    has_main: bool = False
    safe_to_run: bool = False
    run_ok: bool | None = None
    error: str | None = None


def check_script(module_name: str, run: bool = False) -> CheckResult:
    """Import a pipeline script and optionally execute its safe `main` function."""
    result = CheckResult(name=module_name)
    full_name = f"scripts.{module_name}"
    try:
        mod = importlib.import_module(full_name)
        result.imported = True
        result.has_main = callable(getattr(mod, "main", None))
        result.safe_to_run = bool(getattr(mod, "SMOKE_TEST_SAFE", False))

        if run:
            if result.safe_to_run and result.has_main:
                try:
                    mod.main()
                    result.run_ok = True
                except Exception as exc:  # pragma: no cover - surface execution issues
                    result.run_ok = False
                    result.error = str(exc)
            else:
                result.run_ok = None  # skipped

    except Exception as exc:  # pragma: no cover - import problems should be visible
        result.error = str(exc)

    return result


def main(argv: list[str] | None = None) -> int:
    """Run smoke tests for pipeline script importability and optional safe execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test pipeline scripts")
    parser.add_argument("--run", action="store_true", help="Execute safe mains")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO)

    results: List[CheckResult] = []
    for name in SCRIPTS_TO_CHECK:
        logger.info("Checking scripts.%s", name)
        res = check_script(name, run=args.run)
        results.append(res)

    # Print a concise table
    ok = 0
    for r in results:
        status = "OK" if r.imported else "IMPORT FAIL"
        main_s = "main" if r.has_main else "no-main"
        safe_s = "safe" if r.safe_to_run else "unsafe"
        run_s = (
            "ran" if r.run_ok is True else "fail" if r.run_ok is False else "skipped"
        )
        logger.info("%s: %s (%s, %s) -> %s", r.name, status, main_s, safe_s, run_s)
        if r.imported and (r.run_ok is not False):
            ok += 1

    total = len(results)
    logger.info("Smoke-check finished: %d/%d OK (imports + safe runs)", ok, total)

    # non-zero exit when import failures or run failures
    failures = [r for r in results if (not r.imported) or (r.run_ok is False)]
    if failures:
        for f in failures:
            logger.error("Failure: %s -> %s", f.name, f.error)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
