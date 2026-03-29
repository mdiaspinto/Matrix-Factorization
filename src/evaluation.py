"""
Evaluation metrics for implicit feedback recommendation.

Uses the leave-one-out protocol from the eALS paper:
    For each test user, rank the held-out item against all items the user
    has NOT interacted with in training. Compute Hit Rate and NDCG at K.
"""

import numpy as np


def evaluate(P, Q, train_matrix, test_dict, top_k=10):
    """
    Evaluate recommendation quality using Hit Rate @ K and NDCG @ K.

    Args:
        P: np.ndarray (num_users × k) — user factors.
        Q: np.ndarray (num_items × k) — item factors.
        train_matrix: scipy.sparse.csr_matrix — training interactions.
        test_dict: dict {user_id: test_item_id} — one held-out item per user.
        top_k: Cutoff K for ranking metrics.

    Returns:
        dict with "HR@K" and "NDCG@K".
    """
    hits = 0
    ndcg_sum = 0.0
    num_test_users = len(test_dict)

    R_csr = train_matrix.tocsr()
    num_items = Q.shape[0]

    for u, test_item in test_dict.items():
        # Score all items for this user
        scores = Q @ P[u]  # (num_items,)

        # Mask out training items (set to -inf so they don't appear in top-K)
        train_items = R_csr[u].indices
        scores[train_items] = -np.inf

        # Get top-K item indices
        top_k_items = np.argpartition(scores, -top_k)[-top_k:]
        top_k_items = top_k_items[np.argsort(-scores[top_k_items])]

        # Check if test item is in top-K
        if test_item in top_k_items:
            hits += 1
            # NDCG: position of test item in the ranked list (1-indexed)
            rank = np.where(top_k_items == test_item)[0][0] + 1
            ndcg_sum += 1.0 / np.log2(rank + 1)

    hr = hits / num_test_users
    ndcg = ndcg_sum / num_test_users

    return {f"HR@{top_k}": hr, f"NDCG@{top_k}": ndcg}


def evaluate_multiple_k(P, Q, train_matrix, test_dict, k_values=(5, 10, 20)):
    """
    Evaluate at multiple K values in a single pass.

    Returns:
        dict mapping metric names to values, e.g. {"HR@5": 0.3, "NDCG@5": 0.15, ...}
    """
    max_k = max(k_values)
    results = {f"HR@{k}": 0.0 for k in k_values}
    results.update({f"NDCG@{k}": 0.0 for k in k_values})

    R_csr = train_matrix.tocsr()
    num_test_users = len(test_dict)

    for u, test_item in test_dict.items():
        scores = Q @ P[u]
        train_items = R_csr[u].indices
        scores[train_items] = -np.inf

        top_items = np.argpartition(scores, -max_k)[-max_k:]
        top_items = top_items[np.argsort(-scores[top_items])]

        if test_item in top_items:
            rank = np.where(top_items == test_item)[0][0] + 1
            for k in k_values:
                if rank <= k:
                    results[f"HR@{k}"] += 1.0
                    results[f"NDCG@{k}"] += 1.0 / np.log2(rank + 1)

    for key in results:
        results[key] /= num_test_users

    return results


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_and_prepare
    from eals_numpy import eals_train

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    data = load_and_prepare(data_dir, dataset="100k")

    print("\nTraining eALS for evaluation demo...")
    P, Q, losses = eals_train(
        data["train_matrix"],
        num_factors=32,
        num_iter=20,
        reg=0.01,
        alpha=1.0,
        c0=1.0,
    )

    print("\nEvaluating...")
    metrics = evaluate_multiple_k(
        P, Q, data["train_matrix"], data["test_dict"],
        k_values=(5, 10, 20)
    )
    for name, val in sorted(metrics.items()):
        print(f"  {name}: {val:.4f}")
