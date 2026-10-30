"""
shared/adapters.py
Thin adapter functions to fix mismatches between model interfaces without editing model files.
"""
import numpy as np


def normalize_embedding(embedding):
    """L2-normalise a single embedding vector."""
    norm = np.linalg.norm(embedding)
    if norm < 1e-8:
        return embedding
    return embedding / norm


def normalize_embeddings(embeddings):
    """L2-normalise rows of an (N, D) embedding matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return embeddings / norms


def pad_or_truncate_embedding(embedding, target_dim):
    """Pad with zeros or truncate to match target_dim."""
    if len(embedding) == target_dim:
        return embedding
    if len(embedding) > target_dim:
        return embedding[:target_dim]
    padded = np.zeros(target_dim, dtype=embedding.dtype)
    padded[:len(embedding)] = embedding
    return padded


def aggregate_embeddings(embeddings, weights=None):
    """Weighted average of embeddings, then L2-normalise.
    Args:
        embeddings: (N, D) array
        weights: optional (N,) array (e.g., confidence scores)
    Returns:
        (D,) L2-normalised vector
    """
    if len(embeddings) == 0:
        raise ValueError("No embeddings to aggregate")
    if weights is not None:
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.sum()
        avg = np.average(embeddings, axis=0, weights=weights)
    else:
        avg = np.mean(embeddings, axis=0)
    return normalize_embedding(avg)


def confidence_weighted_vote(predictions, key, conf_key):
    """Aggregate attribute predictions by confidence-weighted voting.
    Args:
        predictions: list of dicts with key and conf_key fields
        key: attribute name (e.g., 'colour', 'type')
        conf_key: confidence name (e.g., 'colour_conf', 'type_conf')
    Returns:
        (winner, aggregated_confidence) or (None, 0.0)
    """
    if not predictions:
        return None, 0.0
    from collections import defaultdict
    votes = defaultdict(float)
    for pred in predictions:
        val = pred.get(key)
        conf = pred.get(conf_key, 0.0)
        if val is not None:
            votes[val] += conf
    if not votes:
        return None, 0.0
    winner = max(votes, key=votes.get)
    total_conf = sum(votes.values())
    return winner, votes[winner] / total_conf if total_conf > 0 else 0.0

