"""
Element-wise ALS (eALS) for Implicit Feedback — PySpark RDD implementation.

Parallelization strategy (from the paper, Section 4):
    - When updating P (user factors): partition users across workers.
      Broadcast Q and S^q = Q^T Q. Each worker independently updates
      its assigned users' factor vectors.
    - When updating Q (item factors): partition items across workers.
      Broadcast P and S^p = P^T P. Each worker independently updates
      its assigned items' factor vectors.

This is embarrassingly parallel per half-step because each user's (or item's)
update only depends on the *current* opposite factor matrix, not on other
users (or items) being updated in the same half-step.
"""

import time
import numpy as np
from scipy import sparse


def eals_train_rdd(sc, train_matrix, num_factors=64, num_iter=50,
                   reg=0.01, alpha=1.0, c0=1.0, num_partitions=None,
                   seed=42, verbose=True):
    """
    Train eALS model using PySpark RDDs.

    Args:
        sc: SparkContext.
        train_matrix: scipy.sparse.csr_matrix (num_users × num_items), binary.
        num_factors: Latent dimension k.
        num_iter: Number of ALS iterations.
        reg: L2 regularization lambda.
        alpha: Confidence boost for observed entries (c_ui = 1 + alpha).
        c0: Uniform confidence weight for missing entries.
        num_partitions: Number of RDD partitions (default: sc.defaultParallelism).
        seed: Random seed.
        verbose: Print progress.

    Returns:
        P: np.ndarray (num_users × k).
        Q: np.ndarray (num_items × k).
        losses: list of loss per iteration.
    """
    rng = np.random.RandomState(seed)
    M, N = train_matrix.shape
    k = num_factors

    if num_partitions is None:
        num_partitions = sc.defaultParallelism

    c_obs = 1.0 + alpha

    # Initialize factors
    P = rng.normal(0, 0.01, (M, k))
    Q = rng.normal(0, 0.01, (N, k))

    # Build per-user interaction data: list of (user_id, item_indices_array)
    R_csr = train_matrix.tocsr()
    user_data = []
    for u in range(M):
        items = R_csr[u].indices.copy()
        if len(items) > 0:
            user_data.append((u, items))

    # Build per-item interaction data
    R_csc = train_matrix.tocsc()
    item_data = []
    for i in range(N):
        users = R_csc[:, i].indices.copy()
        if len(users) > 0:
            item_data.append((i, users))

    # Create and cache RDDs (static structure, reused every iteration)
    user_rdd = sc.parallelize(user_data, num_partitions).cache()
    item_rdd = sc.parallelize(item_data, num_partitions).cache()

    losses = []

    for iteration in range(num_iter):
        t_start = time.time()

        # ---- Update P (user factors) ----
        Sq = Q.T @ Q  # k × k
        Q_bc = sc.broadcast(Q)
        Sq_bc = sc.broadcast(Sq)

        def update_user_partition(iterator):
            """Process a partition of users, updating all their factors."""
            q_local = Q_bc.value
            sq_local = Sq_bc.value
            results = []

            for u, item_indices in iterator:
                p_u = P_bc.value[u].copy()
                Q_u = q_local[item_indices]  # |I_u| × k
                pred_cache = Q_u @ p_u

                for f in range(k):
                    q_f = Q_u[:, f]
                    hat_r = 1.0 - pred_cache + p_u[f] * q_f

                    numer = (c_obs - c0) * np.dot(q_f, hat_r)
                    numer -= c0 * (p_u @ sq_local[:, f] - p_u[f] * sq_local[f, f])

                    denom = (c_obs - c0) * np.dot(q_f, q_f) + c0 * sq_local[f, f] + reg

                    old_val = p_u[f]
                    p_u[f] = numer / denom
                    pred_cache += (p_u[f] - old_val) * q_f

                results.append((u, p_u))
            return iter(results)

        P_bc = sc.broadcast(P)
        updated_users = user_rdd.mapPartitions(update_user_partition).collect()
        for u, p_u in updated_users:
            P[u] = p_u

        Q_bc.unpersist()
        Sq_bc.unpersist()
        P_bc.unpersist()

        # ---- Update Q (item factors) ----
        Sp = P.T @ P  # k × k
        P_bc = sc.broadcast(P)
        Sp_bc = sc.broadcast(Sp)

        def update_item_partition(iterator):
            """Process a partition of items, updating all their factors."""
            p_local = P_bc.value
            sp_local = Sp_bc.value
            results = []

            for i, user_indices in iterator:
                q_i = Q_bc2.value[i].copy()
                P_i = p_local[user_indices]
                pred_cache = P_i @ q_i

                for f in range(k):
                    p_f = P_i[:, f]
                    hat_r = 1.0 - pred_cache + q_i[f] * p_f

                    numer = (c_obs - c0) * np.dot(p_f, hat_r)
                    numer -= c0 * (q_i @ sp_local[:, f] - q_i[f] * sp_local[f, f])

                    denom = (c_obs - c0) * np.dot(p_f, p_f) + c0 * sp_local[f, f] + reg

                    old_val = q_i[f]
                    q_i[f] = numer / denom
                    pred_cache += (q_i[f] - old_val) * p_f

                results.append((i, q_i))
            return iter(results)

        Q_bc2 = sc.broadcast(Q)
        updated_items = item_rdd.mapPartitions(update_item_partition).collect()
        for i, q_i in updated_items:
            Q[i] = q_i

        P_bc.unpersist()
        Sp_bc.unpersist()
        Q_bc2.unpersist()

        elapsed = time.time() - t_start

        loss = _compute_loss_fast(P, Q, R_csr, c_obs, c0, reg)
        losses.append(loss)
        if verbose:
            print(f"Iteration {iteration + 1}/{num_iter} | "
                  f"loss={loss:.4f} | time={elapsed:.2f}s")

    # Clean up cached RDDs
    user_rdd.unpersist()
    item_rdd.unpersist()

    return P, Q, losses


def _compute_loss_fast(P, Q, R_csr, c_obs, c0, reg):
    """
    Compute weighted squared loss using the trace trick to avoid O(M*N) cost.

    L = c0 * trace(P^T P @ Q^T Q)                        [all-pairs, r=0 assumed]
      - c0 * sum_{(u,i) observed} (p_u^T q_i)^2          [remove wrong c0 term]
      + c_obs * sum_{(u,i) observed} (1 - p_u^T q_i)^2   [add correct observed term]
      + reg * (||P||^2 + ||Q||^2)
    """
    M = P.shape[0]
    Sp = P.T @ P
    Sq = Q.T @ Q

    loss = c0 * np.trace(Sp @ Sq)

    for u in range(M):
        items = R_csr[u].indices
        if len(items) == 0:
            continue
        preds = Q[items] @ P[u]
        loss -= c0 * np.dot(preds, preds)
        residuals = 1.0 - preds
        loss += c_obs * np.dot(residuals, residuals)

    loss += reg * (np.sum(P ** 2) + np.sum(Q ** 2))
    return loss


if __name__ == "__main__":
    import os
    import sys
    from pyspark import SparkContext, SparkConf

    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_and_prepare

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    data = load_and_prepare(data_dir, dataset="100k")

    conf = SparkConf().setAppName("eALS-RDD").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    print("\nTraining eALS (PySpark RDD)...")
    P, Q, losses = eals_train_rdd(
        sc,
        data["train_matrix"],
        num_factors=32,
        num_iter=10,
        reg=0.01,
        alpha=1.0,
        c0=1.0,
    )

    print(f"\nFinal loss: {losses[-1]:.4f}")
    print(f"P shape: {P.shape}, Q shape: {Q.shape}")

    sc.stop()
