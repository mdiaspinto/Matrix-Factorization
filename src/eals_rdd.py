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
                   reg=0.01, alpha=1.0, c0=1.0, pop_alpha=0.5,
                   num_partitions=None, seed=42, verbose=True):
    """
    Train eALS model using PySpark RDDs with popularity-aware missing data weighting.

    Missing entry (u, i) receives weight ci proportional to item i's popularity
    raised to pop_alpha (paper Section 4.1, Eq. 8). Setting pop_alpha=0 recovers
    uniform weighting (ci = c0/N for all items).

    Args:
        sc: SparkContext.
        train_matrix: scipy.sparse.csr_matrix (num_users × num_items), binary.
        num_factors: Latent dimension k.
        num_iter: Number of ALS iterations.
        reg: L2 regularization lambda.
        alpha: Confidence boost for observed entries (c_ui = 1 + alpha).
        c0: Overall scale of missing data weights (paper's c0).
        pop_alpha: Popularity exponent (paper's alpha, default 0.5).
                   pop_alpha=0 gives uniform weights.
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

    # Per-item popularity weights (paper Eq. 8):
    #   ci = c0 * f_i^pop_alpha / sum_j(f_j^pop_alpha)
    # where f_i = |R_i| / sum_j|R_j|  (item interaction frequency)
    # pop_alpha=0 → uniform weights (ci = c0/N); pop_alpha=0.5 is paper default.
    item_counts = np.array(train_matrix.sum(axis=0)).flatten()  # shape: (N,)
    f = item_counts / item_counts.sum()
    f_alpha = np.power(f, pop_alpha)
    c_items = c0 * f_alpha / f_alpha.sum()                      # shape: (N,)

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

    # Build per-item interaction data: (item_id, user_indices, ci)
    R_csc = train_matrix.tocsc()
    item_data = []
    for i in range(N):
        users = R_csc[:, i].indices.copy()
        if len(users) > 0:
            item_data.append((i, users, float(c_items[i])))

    # Create and cache RDDs (static structure, reused every iteration)
    user_rdd = sc.parallelize(user_data, num_partitions).cache()
    item_rdd = sc.parallelize(item_data, num_partitions).cache()

    losses = []

    for iteration in range(num_iter):
        t_start = time.time()

        # ---- Update P (user factors) ----
        # S^q is weighted by item popularity: S^q = sum_i ci * qi * qi^T
        Sq = Q.T @ (Q * c_items[:, np.newaxis])  # k × k, weighted
        Q_bc = sc.broadcast(Q)
        Sq_bc = sc.broadcast(Sq)
        c_items_bc = sc.broadcast(c_items)

        def update_user_partition(iterator):
            """Process a partition of users, updating all their factors."""
            q_local = Q_bc.value
            sq_local = Sq_bc.value
            c_local = c_items_bc.value
            results = []

            for u, item_indices in iterator:
                p_u = P_bc.value[u].copy()
                Q_u = q_local[item_indices]       # |I_u| × k
                c_u = c_local[item_indices]       # |I_u|, ci per observed item
                pred_cache = Q_u @ p_u

                for f in range(k):
                    q_f = Q_u[:, f]
                    hat_r = pred_cache - p_u[f] * q_f  # r̂^f_ui for each observed item

                    # obs_pull: sum_{i in Iu} [c_obs - (c_obs-ci)*r̂^f_ui] * qif
                    # c_u replaces the scalar c0 — each item has its own weight
                    obs_pull = c_obs * np.sum(q_f) - np.dot((c_obs - c_u) * hat_r, q_f)

                    # missing_pull uses weighted Sq — c0 is absorbed into Sq already
                    missing_pull = p_u @ sq_local[:, f] - p_u[f] * sq_local[f, f]

                    numer = obs_pull - missing_pull

                    # denom: sum_{i in Iu} (c_obs-ci)*qif^2 + Sq[f,f] + lambda
                    denom = np.dot((c_obs - c_u) * q_f, q_f) + sq_local[f, f] + reg

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
        c_items_bc.unpersist()

        # ---- Update Q (item factors) ----
        # S^p = P^T P stays unweighted — ci is per-item, not per-user
        Sp = P.T @ P  # k × k
        P_bc = sc.broadcast(P)
        Sp_bc = sc.broadcast(Sp)

        def update_item_partition(iterator):
            """Process a partition of items, updating all their factors."""
            p_local = P_bc.value
            sp_local = Sp_bc.value
            results = []

            for i, user_indices, ci in iterator:  # ci is this item's popularity weight
                q_i = Q_bc2.value[i].copy()
                P_i = p_local[user_indices]
                pred_cache = P_i @ q_i

                for f in range(k):
                    p_f = P_i[:, f]
                    hat_r = pred_cache - q_i[f] * p_f  # r̂^f_ui for each observed user

                    # obs_pull: sum_{u in Ri} [c_obs - (c_obs-ci)*r̂^f_ui] * puf
                    # ci is scalar — same weight for all users of this item
                    obs_pull = c_obs * np.sum(p_f) - (c_obs - ci) * np.dot(p_f, hat_r)

                    # missing_pull: ci replaces c0 (this item's own weight)
                    missing_pull = ci * (q_i @ sp_local[:, f] - q_i[f] * sp_local[f, f])

                    numer = obs_pull - missing_pull
                    denom = (c_obs - ci) * np.dot(p_f, p_f) + ci * sp_local[f, f] + reg

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

        loss = _compute_loss_fast(P, Q, R_csr, c_obs, c_items, reg)
        losses.append(loss)
        if verbose:
            print(f"Iteration {iteration + 1}/{num_iter} | "
                  f"loss={loss:.4f} | time={elapsed:.2f}s")

    # Clean up cached RDDs
    user_rdd.unpersist()
    item_rdd.unpersist()

    return P, Q, losses


def _compute_loss_fast(P, Q, R_csr, c_obs, c_items, reg):
    """
    Compute weighted squared loss (paper Eq. 7) efficiently.

    Decomposition:
      1. All-pairs term (r=0, weight=ci):
            sum_u sum_i ci*(p_u^T q_i)^2 = trace(Sp @ Sq_weighted)
         where Sq_weighted = Q^T @ diag(c_items) @ Q
      2. Correct observed entries: subtract ci*pred^2, add c_obs*(1-pred)^2
      3. Add regularization.

    Cost: O(k^3 + |R|*k).
    """
    M = P.shape[0]
    Sp = P.T @ P
    Sq = Q.T @ (Q * c_items[:, np.newaxis])  # weighted by ci

    loss = np.trace(Sp @ Sq)

    for u in range(M):
        items = R_csr[u].indices
        if len(items) == 0:
            continue
        preds = Q[items] @ P[u]
        loss -= np.dot(c_items[items], preds * preds)  # remove ci*pred^2
        residuals = 1.0 - preds
        loss += c_obs * np.dot(residuals, residuals)   # add c_obs*(1-pred)^2

    loss += reg * (np.sum(P ** 2) + np.sum(Q ** 2))
    return loss


if __name__ == "__main__":
    import os
    import sys
    from pyspark import SparkContext, SparkConf

    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_and_prepare

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    data = load_and_prepare(data_dir)

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
