"""
Evaluation module for copyright compliance assessment.
"""

from .metrics import (
    calculate_rouge_scores,
    calculate_lcs,
    calculate_trans_cos_sim,
    check_for_rejection
)

from .evaluator import (
    evaluate_responses,
    calculate_metrics_mean
)

__all__ = [
    'calculate_rouge_scores',
    'calculate_lcs',
    'calculate_trans_cos_sim',
    'check_for_rejection',
    'evaluate_responses',
    'calculate_metrics_mean'
]



