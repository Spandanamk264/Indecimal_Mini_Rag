"""
Evaluation Module - __init__.py
================================

Evaluation and metrics for RAG system quality.
"""

from .evaluator import (
    RetrievalEvalResult,
    GenerationEvalResult,
    EvalExample,
    EvalDataset,
    RetrievalEvaluator,
    GenerationEvaluator,
    RAGEvaluator,
    create_sample_eval_dataset
)

__all__ = [
    "RetrievalEvalResult",
    "GenerationEvalResult",
    "EvalExample",
    "EvalDataset",
    "RetrievalEvaluator",
    "GenerationEvaluator",
    "RAGEvaluator",
    "create_sample_eval_dataset"
]
