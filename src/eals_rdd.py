"""
Element-wise ALS (eALS) for Implicit Feedback — PySpark RDD implementation.

Parallelization strategy (from the paper, Section 4):
    - When updating P (user factors): partition users across workers.
      Broadcast Q and S^q = sum_i ci * qi * qi^T. Each worker independently
      updates its assigned users' factor vectors.
    - When updating Q (item factors): partition items across workers.
      Broadcast P and S^p = P^T P. Each worker independently updates
      its assigned items' factor vectors.

This is embarrassingly parallel per half-step because each user's (or item's)
update only depends on the *current* opposite factor matrix, not on other
users (or items) being updated in the same half-step.

Variable naming follows the paper (He et al., SIGIR 2016):
    K       : number of latent factors
    lam     : regularisation lambda (lambda is a Python keyword)
    w_obs   : confidence weight for observed entries (paper: w_ui = 1, so w_obs=0)
    c0      : overall scale of missing data weights (paper: c0)
    alpha   : popularity exponent for missing data weighting (paper: alpha, Eq. 8)
    ci      : per-item missing data weight (paper: ci)
    S^q     : item cache matrix = sum_i ci * qi * qi^T  (code: Sq)
    S^p     : user cache matrix = P^T P                 (code: Sp)
    r̂^f_ui : prediction for (u,i) with factor f removed (code: hat_r)
"""

import time
import numpy as np
from scipy import sparse


def eals_train_rdd(sc, train_matrix, K=128, num_iter=50,
                   lam=0.01, w_obs=0.0, c0=512, alpha=0.4,
                   num_partitions=None, seed=42, verbose=True):
    """
    Train eALS model using PySpark RDDs with popularity-aware missing data weighting.

    Default hyperparameters match the paper's optimal settings for Yelp
    (He et al., SIGIR 2016, Section 5.2.1 and Figures 1a/1b).

    Missing entry (u, i) receives weight ci proportional to item i's popularity
    raised to alpha (paper Section 4.1, Eq. 8). Setting alpha=0 recovers
    uniform weighting (ci = c0/N for all items).

    Args:
        sc            : SparkContext.
        train_matrix  : scipy.sparse.csr_matrix (num_users x num_items), binary.
        K             : Number of latent factors (paper: K, default 128).
        num_iter      : Number of ALS iterations.
        lam           : L2 regularisation lambda (paper: lambda, default 0.01).
        w_obs         : Confidence weight for observed entries beyond 1.
                        c_obs = 1 + w_obs. Paper sets w_ui = 1 so w_obs = 0.
        c0            : Overall scale of missing data weights (paper: c0).
                        Default 512 — paper's optimal value for Yelp.
        alpha         : Popularity exponent (paper: alpha, default 0.4).
                        Controls how strongly popular items are weighted as
                        negatives. alpha=0 gives uniform weights.
        num_partitions: Number of RDD partitions (default: sc.defaultParallelism).
        seed          : Random seed.
        verbose       : Print progress.

    Returns:
        P     : np.ndarray (num_users x K) — user latent factor matrix.
        Q     : np.ndarray (num_items x K) — item latent factor matrix.
        losses: list of training loss per iteration.
    """
    rng = np.random.RandomState(seed)
    M, N = train_matrix.shape

    if num_partitions is None:
        num_partitions = sc.defaultParallelism

    # Confidence for observed entries: w_ui in paper = 1 + w_obs
    # Paper sets w_ui = 1 for all observed, so w_obs defaults to 0.
    c_obs = 1.0 + w_obs

    # Per-item popularity weights ci (paper Eq. 8):
    #   ci = c0 * f_i^alpha / sum_j(f_j^alpha)
    # where f_i = |R_i| / sum_j|R_j|  (item interaction frequency)
    # alpha=0 -> uniform weights (ci = c0/N); paper default alpha=0.4 for Yelp.
    item_counts = np.array(train_matrix.sum(axis=0)).flatten()  # shape: (N,)
    f       = item_counts / item_counts.sum()
    f_alpha = np.power(f, alpha)
    c_items = c0 * f_alpha / f_alpha.sum()                      # shape: (N,)

    # Initialise latent factor matrices
    P = rng.normal(0, 0.01, (M, K))
    Q = rng.normal(0, 0.01, (N, K))

    # Build per-user interaction data: list of (user_id, item_indices)
    R_csr = train_matrix.tocsr()
    user_data = []
    for u in range(M):
        items = R_csr[u].indices.copy()
        if len(items) > 0:
            user_data.append((u, items))

    # Build per-item interaction data: (item_id, user_indices, ci)
    # ci is embedded in the tuple so each worker has it without an extra broadcast.
    R_csc = train_matrix.tocsc()
    item_data = []
    for i in range(N):
        users = R_csc[:, i].indices.copy()
        if len(users) > 0:
            item_data.append((i, users, float(c_items[i])))

    # Cache RDDs — structure is static across all iterations
    user_rdd = sc.parallelize(user_data, num_partitions).cache()
    item_rdd = sc.parallelize(item_data, num_partitions).cache()

    losses = []

    for iteration in range(num_iter):
        t_start = time.time()

        # ---- Update P (user factors) ----
        # S^q = sum_i ci * qi * qi^T  (paper Section 4.2, weighted item cache)
        Sq         = Q.T @ (Q * c_items[:, np.newaxis])  # K x K
        Q_bc       = sc.broadcast(Q)
        Sq_bc      = sc.broadcast(Sq)
        c_items_bc = sc.broadcast(c_items)

        def update_user_partition(iterator):
            """Update all user factor vectors in this partition (paper Eq. 12)."""
            Q_local       = Q_bc.value
            Sq_local      = Sq_bc.value
            c_items_local = c_items_bc.value
            results       = []

            for u, item_indices in iterator:
                p_u        = P_bc.value[u].copy()
                Q_u        = Q_local[item_indices]        # |I_u| x K
                c_u        = c_items_local[item_indices]  # ci per observed item
                pred_cache = Q_u @ p_u                    # r̂_ui for observed items

                for f in range(K):
                    q_f   = Q_u[:, f]
                    # r̂^(f)_ui = r̂_ui - p_uf * q_if  (paper notation)
                    hat_r = pred_cache - p_u[f] * q_f

                    # Numerator (paper Eq. 12, w_ui = c_obs, r_ui = 1):
                    #   sum_{i in Iu} [c_obs - (c_obs-ci) * r̂^f_ui] * q_if
                    #   - (p_u @ S^q[:,f] - p_uf * S^q[f,f])
                    obs_pull     = c_obs * np.sum(q_f) - np.dot((c_obs - c_u) * hat_r, q_f)
                    missing_pull = p_u @ Sq_local[:, f] - p_u[f] * Sq_local[f, f]
                    numer        = obs_pull - missing_pull

                    # Denominator (paper Eq. 12):
                    #   sum_{i in Iu} (c_obs-ci) * q_if^2 + S^q[f,f] + lambda
                    denom = np.dot((c_obs - c_u) * q_f, q_f) + Sq_local[f, f] + lam

                    old_val    = p_u[f]
                    p_u[f]     = numer / denom
                    pred_cache += (p_u[f] - old_val) * q_f  # O(|Iu|) cache patch

                results.append((u, p_u))
            return iter(results)

        P_bc          = sc.broadcast(P)
        updated_users = user_rdd.mapPartitions(update_user_partition).collect()
        for u, p_u in updated_users:
            P[u] = p_u

        Q_bc.unpersist()
        Sq_bc.unpersist()
        P_bc.unpersist()
        c_items_bc.unpersist()

        # ---- Update Q (item factors) ----
        # S^p = P^T P  (unweighted — ci is per-item, not per-user)
        Sp    = P.T @ P  # K x K
        P_bc  = sc.broadcast(P)
        Sp_bc = sc.broadcast(Sp)

        def update_item_partition(iterator):
            """Update all item factor vectors in this partition (paper Eq. 13)."""
            P_local  = P_bc.value
            Sp_local = Sp_bc.value
            results  = []

            for i, user_indices, ci in iterator:  # ci: this item's popularity weight
                q_i        = Q_bc2.value[i].copy()
                P_i        = P_local[user_indices]    # |R_i| x K
                pred_cache = P_i @ q_i                # r̂_ui for observed users

                for f in range(K):
                    p_f   = P_i[:, f]
                    # r̂^(f)_ui = r̂_ui - q_if * p_uf  (paper notation)
                    hat_r = pred_cache - q_i[f] * p_f

                    # Numerator (paper Eq. 13, ci scalar for this item):
                    #   sum_{u in Ri} [c_obs - (c_obs-ci) * r̂^f_ui] * p_uf
                    #   - ci * (q_i @ S^p[:,f] - q_if * S^p[f,f])
                    obs_pull     = c_obs * np.sum(p_f) - (c_obs - ci) * np.dot(p_f, hat_r)
                    missing_pull = ci * (q_i @ Sp_local[:, f] - q_i[f] * Sp_local[f, f])
                    numer        = obs_pull - missing_pull

                    # Denominator (paper Eq. 13):
                    #   sum_{u in Ri} (c_obs-ci) * p_uf^2 + ci * S^p[f,f] + lambda
                    denom = (c_obs - ci) * np.dot(p_f, p_f) + ci * Sp_local[f, f] + lam

                    old_val    = q_i[f]
                    q_i[f]     = numer / denom
                    pred_cache += (q_i[f] - old_val) * p_f  # O(|Ri|) cache patch

                results.append((i, q_i))
            return iter(results)

        Q_bc2         = sc.broadcast(Q)
        updated_items = item_rdd.mapPartitions(update_item_partition).collect()
        for i, q_i in updated_items:
            Q[i] = q_i

        P_bc.unpersist()
        Sp_bc.unpersist()
        Q_bc2.unpersist()

        elapsed = time.time() - t_start

        loss = _compute_loss(P, Q, R_csr, c_obs, c_items, lam)
        losses.append(loss)
        if verbose:
            print(f"Iteration {iteration + 1}/{num_iter} | "
                  f"loss={loss:.4f} | time={elapsed:.2f}s")

    user_rdd.unpersist()
    item_rdd.unpersist()

    return P, Q, losses


def _compute_loss(P, Q, R_csr, c_obs, c_items, lam):
    """
    Compute the weighted squared loss (paper Eq. 7) efficiently.

    Uses the trace trick to avoid iterating over all M*N pairs:

      L = trace(S^p @ S^q)                            [all-pairs, r=0, weight=ci]
        - sum_{(u,i) in R} ci * (p_u^T q_i)^2        [remove wrong ci term]
        + c_obs * sum_{(u,i) in R} (1 - p_u^T q_i)^2 [add correct observed term]
        + lambda * (||P||^2 + ||Q||^2)

    where S^q = Q^T diag(c_items) Q.
    Cost: O(K^3 + |R| K) instead of O(M N K).
    """
    M  = P.shape[0]
    Sp = P.T @ P
    Sq = Q.T @ (Q * c_items[:, np.newaxis])  # weighted S^q

    loss = np.trace(Sp @ Sq)

    for u in range(M):
        items = R_csr[u].indices
        if len(items) == 0:
            continue
        preds  = Q[items] @ P[u]
        ci_u   = c_items[items]
        loss  -= np.dot(ci_u, preds * preds)
        loss  += c_obs * np.dot(1.0 - preds, 1.0 - preds)

    loss += lam * (np.sum(P ** 2) + np.sum(Q ** 2))
    return loss


if __name__ == "__main__":
    import os
    import sys
    from pyspark import SparkContext, SparkConf

    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_and_prepare

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    data     = load_and_prepare(data_dir)

    conf = SparkConf().setAppName("eALS-RDD").setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    print("\nTraining eALS (PySpark RDD) on Yelp — paper hyperparameters...")
    P, Q, losses = eals_train_rdd(
        sc,
        data["train_matrix"],
        K        = 128,   # paper: K=128 for Yelp
        num_iter = 50,    # paper: run to convergence (~50 sufficient)
        lam      = 0.01,  # paper: lambda=0.01
        w_obs    = 0.0,   # paper: w_ui=1 for observed entries (w_obs=0)
        c0       = 512,   # paper: optimal c0=512 for Yelp (Figure 1a)
        alpha    = 0.4,   # paper: optimal alpha=0.4 for Yelp (Figure 1b)
    )

    print(f"\nFinal loss : {losses[-1]:.4f}")
    print(f"P shape    : {P.shape}")
    print(f"Q shape    : {Q.shape}")

    sc.stop()
