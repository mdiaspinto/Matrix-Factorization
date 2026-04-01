"""
Element-wise ALS (eALS) for Implicit Feedback — NumPy reference implementation.

Based on: "Fast Matrix Factorization for Online Recommendation with Implicit Feedback"
by He et al. (2016).

Key insight: by precomputing S^q = Q^T Q (a small k×k matrix), the cost of each
element-wise update drops from O(N*k) to O(|I_u|*k + k^2), avoiding iteration
over all missing entries.

Notation:
    R         : user-item interaction matrix (M users × N items), binary
    P (M × k) : user latent factor matrix
    Q (N × k) : item latent factor matrix
    c_ui      : confidence for observed interactions = 1 + alpha
    c_0       : confidence for missing (unobserved) entries (uniform weight)
    S^q = Q^T Q : precomputed k×k matrix for efficient missing-data handling
"""

import time
import numpy as np
from scipy import sparse


def eals_train(train_matrix, num_factors=64, num_iter=50,
               reg=0.01, alpha=1.0, c0=1.0, seed=42, verbose=True):
    """
    Train eALS model on a sparse implicit feedback matrix.

    Args:
        train_matrix: scipy.sparse.csr_matrix (num_users × num_items), binary.
        num_factors: Latent dimension k.
        num_iter: Number of ALS iterations.
        reg: L2 regularization lambda.
        alpha: Confidence boost for observed interactions (c_ui = 1 + alpha).
        c0: Uniform confidence weight for missing entries.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        P: np.ndarray (num_users × k) — user factors.
        Q: np.ndarray (num_items × k) — item factors.
        losses: list of training loss per iteration.
    """
    rng = np.random.RandomState(seed)
    M, N = train_matrix.shape
    k = num_factors

    # Initialize factors with small random values
    P = rng.normal(0, 0.01, (M, k))
    Q = rng.normal(0, 0.01, (N, k))

    # Confidence for observed entries
    c_obs = 1.0 + alpha  # since all r_ui = 1 after binarization

    # Precompute per-user and per-item interaction lists (CSR/CSC)
    R_csr = train_matrix.tocsr()
    R_csc = train_matrix.tocsc()

    losses = []

    for iteration in range(num_iter):
        t_start = time.time()

        # --- Update user factors P ---
        Sq = Q.T @ Q  # k × k
        _update_users(P, Q, R_csr, Sq, c_obs, c0, reg, M, k)

        # --- Update item factors Q ---
        Sp = P.T @ P  # k × k
        _update_items(P, Q, R_csc, Sp, c_obs, c0, reg, N, k)

        elapsed = time.time() - t_start

        loss = _compute_loss(P, Q, R_csr, c_obs, c0, reg, M)
        losses.append(loss)
        if verbose:
            print(f"Iteration {iteration + 1}/{num_iter} | "
                  f"loss={loss:.4f} | time={elapsed:.2f}s")

    return P, Q, losses


def _update_users(P, Q, R_csr, Sq, c_obs, c0, reg, M, k):
    """
    Update all user factors element-wise.

    For user u, factor f, the closed-form update is:

        p_uf = numerator / denominator

    numerator = sum_{i in I_u} (c_ui - c0) * q_if * hat_r_ui_f
                - c0 * (p_u^T @ Sq[:, f] - p_uf * Sq[f, f])

    where hat_r_ui_f = 1 - sum_{f'!=f} p_uf' * q_if'
                     = 1 - (p_u @ q_i - p_uf * q_if)

    denominator = sum_{i in I_u} (c_ui - c0) * q_if^2
                  + c0 * Sq[f, f] + lambda
    """
    for u in range(M):
        # Items that user u interacted with
        item_indices = R_csr[u].indices

        if len(item_indices) == 0:
            continue

        # Q submatrix for observed items: |I_u| × k
        Q_u = Q[item_indices]

        # Precompute prediction cache: p_u @ q_i for each observed item
        pred_cache = Q_u @ P[u]  # |I_u|

        for f in range(k):
            # Partial residual: remove contribution of factor f
            # hat_r = 1 - (pred - p_uf * q_if) = 1 - pred + p_uf * q_if
            q_f = Q_u[:, f]  # |I_u|
            hat_r = 1.0 - pred_cache + P[u, f] * q_f

            # Numerator
            numer = (c_obs - c0) * np.dot(q_f, hat_r)
            # Contribution from all items via precomputed Sq
            numer -= c0 * (P[u] @ Sq[:, f] - P[u, f] * Sq[f, f])

            # Denominator
            denom = (c_obs - c0) * np.dot(q_f, q_f) + c0 * Sq[f, f] + reg

            # Update
            old_val = P[u, f]
            P[u, f] = numer / denom

            # Update prediction cache to reflect the change
            pred_cache += (P[u, f] - old_val) * q_f


def _update_items(P, Q, R_csc, Sp, c_obs, c0, reg, N, k):
    """
    Update all item factors element-wise (symmetric to user update).
    """
    for i in range(N):
        # Users who interacted with item i
        user_indices = R_csc[:, i].indices

        if len(user_indices) == 0:
            continue

        P_i = P[user_indices]
        pred_cache = P_i @ Q[i]

        for f in range(k):
            p_f = P_i[:, f]
            hat_r = 1.0 - pred_cache + Q[i, f] * p_f

            numer = (c_obs - c0) * np.dot(p_f, hat_r)
            numer -= c0 * (Q[i] @ Sp[:, f] - Q[i, f] * Sp[f, f])

            denom = (c_obs - c0) * np.dot(p_f, p_f) + c0 * Sp[f, f] + reg

            old_val = Q[i, f]
            Q[i, f] = numer / denom

            pred_cache += (Q[i, f] - old_val) * p_f


def _compute_loss(P, Q, R_csr, c_obs, c0, reg, M):
    """
    Compute the weighted squared loss:
        L = sum_u sum_i c_ui * (r_ui - p_u^T q_i)^2 + lambda*(||P||^2 + ||Q||^2)

    Uses the decomposition trick to avoid iterating over all (u,i) pairs:
        sum_u sum_i c0 * (p_u^T q_i)^2 = c0 * ||P^T Q||_F^2 = c0 * trace(Sp @ Sq)

    Then adjust for observed entries which have different weight and target.
    """
    Sp = P.T @ P  # k × k
    Sq = Q.T @ Q  # k × k

    # Loss from all entries assuming r=0 and weight=c0
    loss = c0 * np.trace(Sp @ Sq)

    # Correct for observed entries
    for u in range(M):
        items = R_csr[u].indices
        if len(items) == 0:
            continue

        Q_u = Q[items]
        preds = Q_u @ P[u]  # predictions for observed items

        # Remove c0 * pred^2 contribution (already counted above)
        loss -= c0 * np.dot(preds, preds)
        # Add correct contribution: c_obs * (1 - pred)^2
        residuals = 1.0 - preds
        loss += c_obs * np.dot(residuals, residuals)

    # Regularization
    loss += reg * (np.sum(P ** 2) + np.sum(Q ** 2))

    return loss


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_and_prepare

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    data = load_and_prepare(data_dir, dataset="100k")

    print("\nTraining eALS (NumPy)...")
    P, Q, losses = eals_train(
        data["train_matrix"],
        num_factors=32,
        num_iter=10,
        reg=0.01,
        alpha=1.0,
        c0=1.0,
    )

    print(f"\nFinal loss: {losses[-1]:.4f}")
    print(f"P shape: {P.shape}, Q shape: {Q.shape}")
