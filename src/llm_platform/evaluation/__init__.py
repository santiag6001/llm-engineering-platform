"""Offline-testable evaluation and regression tooling.

This package is intentionally independent from the serving application.
"""

from llm_platform.evaluation.dataset import load_dataset
from llm_platform.evaluation.runner import EvaluationRunner

__all__ = ["EvaluationRunner", "load_dataset"]
