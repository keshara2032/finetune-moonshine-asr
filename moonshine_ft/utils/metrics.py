"""
Evaluation metrics for ASR models.
"""

import os
import evaluate
import numpy as np
from typing import List, Tuple


def _load_metric(metric_name: str):
    return evaluate.load(
        metric_name,
        cache_dir=os.environ.get("HF_EVALUATE_CACHE")
    )


def compute_wer(predictions: List[str], references: List[str]) -> float:
    """
    Compute Word Error Rate (WER).

    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions

    Returns:
        WER as a percentage (0-100)
    """
    metric = _load_metric('wer')

    # Handle empty strings
    pred_empty = np.array([p.strip() == "" for p in predictions])
    ref_empty = np.array([r.strip() == "" for r in references])

    wer_scores = np.ones(len(predictions))
    wer_scores[pred_empty & ref_empty] = 0  # Both empty = perfect

    # Compute WER for non-empty references
    non_empty = ~ref_empty
    if np.any(non_empty):
        non_empty_wer = metric.compute(
            predictions=np.array(predictions)[non_empty].tolist(),
            references=np.array(references)[non_empty].tolist()
        )
        wer_scores[non_empty] = non_empty_wer

    return 100 * np.mean(wer_scores)


def compute_cer(predictions: List[str], references: List[str]) -> float:
    """
    Compute Character Error Rate (CER).

    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions

    Returns:
        CER as a percentage (0-100)
    """
    metric = _load_metric('cer')

    # Handle empty strings
    pred_empty = np.array([p.strip() == "" for p in predictions])
    ref_empty = np.array([r.strip() == "" for r in references])

    cer_scores = np.ones(len(predictions))
    cer_scores[pred_empty & ref_empty] = 0  # Both empty = perfect

    # Compute CER for non-empty references
    non_empty = ~ref_empty
    if np.any(non_empty):
        non_empty_cer = metric.compute(
            predictions=np.array(predictions)[non_empty].tolist(),
            references=np.array(references)[non_empty].tolist()
        )
        cer_scores[non_empty] = non_empty_cer

    return 100 * np.mean(cer_scores)


def compute_detailed_metrics(
    predictions: List[str],
    references: List[str]
) -> dict:
    """
    Compute detailed evaluation metrics.

    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions

    Returns:
        Dictionary with WER, CER, and per-sample statistics
    """
    wer = compute_wer(predictions, references)
    cer = compute_cer(predictions, references)

    # Compute per-sample WER
    wer_metric = _load_metric('wer')
    individual_wers = []
    for pred, ref in zip(predictions, references):
        if ref.strip() == "" and pred.strip() == "":
            individual_wer = 0.0
        elif ref.strip() == "" or pred.strip() == "":
            individual_wer = 1.0
        else:
            individual_wer = wer_metric.compute(predictions=[pred], references=[ref])
        individual_wers.append(individual_wer)

    return {
        'wer': wer,
        'cer': cer,
        'individual_wers': individual_wers,
        'mean_wer': np.mean(individual_wers) * 100,
        'median_wer': np.median(individual_wers) * 100,
        'std_wer': np.std(individual_wers) * 100,
        'num_samples': len(predictions),
        'perfect_predictions': sum(1 for w in individual_wers if w == 0),
    }
