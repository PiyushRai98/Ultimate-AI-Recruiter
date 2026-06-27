"""Offline evaluation metrics for ranking quality assessment.

Implements standard IR metrics: NDCG, MAP, P@K, MRR.
"""

from __future__ import annotations

import numpy as np
from loguru import logger


def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    """Compute Discounted Cumulative Gain at K.

    Args:
        relevance: Array of relevance scores in rank order.
        k: Cutoff position.

    Returns:
        DCG@K score.
    """
    relevance = np.asarray(relevance)[:k]
    if relevance.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevance.size + 2))
    return float(np.sum(relevance / discounts))


def ndcg_at_k(relevance: np.ndarray, k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at K.

    Args:
        relevance: Array of relevance scores in rank order.
        k: Cutoff position.

    Returns:
        NDCG@K score between 0.0 and 1.0.
    """
    actual_dcg = dcg_at_k(relevance, k)
    ideal_relevance = np.sort(relevance)[::-1]
    ideal_dcg = dcg_at_k(ideal_relevance, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def precision_at_k(relevance: np.ndarray, k: int, threshold: float = 1.0) -> float:
    """Compute Precision at K.

    Args:
        relevance: Array of relevance scores in rank order.
        k: Cutoff position.
        threshold: Minimum relevance to count as positive.

    Returns:
        P@K score.
    """
    relevance = np.asarray(relevance)[:k]
    if relevance.size == 0:
        return 0.0
    return float(np.sum(relevance >= threshold)) / k


def mean_average_precision(relevance: np.ndarray, threshold: float = 1.0) -> float:
    """Compute Mean Average Precision.

    Args:
        relevance: Array of relevance scores in rank order.
        threshold: Minimum relevance to count as positive.

    Returns:
        MAP score.
    """
    relevance = np.asarray(relevance)
    relevant_mask = relevance >= threshold
    n_relevant = int(np.sum(relevant_mask))

    if n_relevant == 0:
        return 0.0

    precisions = []
    relevant_count = 0
    for i, is_relevant in enumerate(relevant_mask):
        if is_relevant:
            relevant_count += 1
            precisions.append(relevant_count / (i + 1))

    return float(np.mean(precisions)) if precisions else 0.0


def mrr(relevance: np.ndarray, threshold: float = 1.0) -> float:
    """Compute Mean Reciprocal Rank.

    Args:
        relevance: Array of relevance scores in rank order.
        threshold: Minimum relevance to count as positive.

    Returns:
        MRR score.
    """
    relevance = np.asarray(relevance)
    for i, score in enumerate(relevance):
        if score >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def compute_competition_score(relevance: np.ndarray) -> dict[str, float]:
    """Compute the competition composite score.

    Final composite = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10

    Args:
        relevance: Array of relevance scores for the top-100 ranked candidates.

    Returns:
        Dictionary with individual metrics and composite score.
    """
    relevance = np.asarray(relevance, dtype=np.float64)

    ndcg10 = ndcg_at_k(relevance, 10)
    ndcg50 = ndcg_at_k(relevance, 50)
    map_score = mean_average_precision(relevance)
    p10 = precision_at_k(relevance, 10)
    p5 = precision_at_k(relevance, 5)
    mrr_score = mrr(relevance)

    composite = 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * map_score + 0.05 * p10

    return {
        "ndcg@10": ndcg10,
        "ndcg@50": ndcg50,
        "map": map_score,
        "p@10": p10,
        "p@5": p5,
        "mrr": mrr_score,
        "composite": composite,
    }


def evaluate_against_pseudo_labels(
    ranked_ids: list[str],
    pseudo_relevance: dict[str, float],
) -> dict[str, float]:
    """Evaluate ranking against pseudo-relevance labels.

    Args:
        ranked_ids: Candidate IDs in rank order (top-100).
        pseudo_relevance: Dict mapping candidate_id -> relevance score.

    Returns:
        Evaluation metrics.
    """
    relevance = np.array(
        [pseudo_relevance.get(cid, 0.0) for cid in ranked_ids],
        dtype=np.float64,
    )

    metrics = compute_competition_score(relevance)
    logger.info(
        f"Evaluation: NDCG@10={metrics['ndcg@10']:.4f}, "
        f"NDCG@50={metrics['ndcg@50']:.4f}, "
        f"MAP={metrics['map']:.4f}, "
        f"P@10={metrics['p@10']:.4f}, "
        f"Composite={metrics['composite']:.4f}"
    )
    return metrics
