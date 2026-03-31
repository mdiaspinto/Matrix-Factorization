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


import time
import numpy as np

import time
import numpy as np

def eals_train_rdd(sc, train_matrix, num_factors=64, num_iter=50,
                   reg=0.01, alpha=1.0, c0=1.0, num_partitions=None,
                   seed=42, verbose=True):
    """
    Train eALS model using PySpark RDDs.
    Features: Distributed MapReduce + Popularity Weights + Checkpointing + Static Topology (Cake Optimization).
    """
    rng = np.random.RandomState(seed)
    M, N = train_matrix.shape
    k = num_factors

    if num_partitions is None:
        num_partitions = sc.defaultParallelism

    c_obs = 1.0 + alpha
    pop_alpha = 0.5

    # ==========================================
    # PRE-COMPUTATION: Item Popularity Weights
    # ==========================================
    coo = train_matrix.tocoo()
    interactions = list(zip(coo.row, coo.col))
    interactions_rdd = sc.parallelize(interactions, num_partitions).cache()

    item_freq_rdd = interactions_rdd.map(lambda x: (x[1], 1)).reduceByKey(lambda a, b: a + b)
    sum_f_alpha = item_freq_rdd.map(lambda x: x[1]**pop_alpha).sum()
    item_weights_rdd = item_freq_rdd.mapValues(lambda f: c0 * (f**pop_alpha) / sum_f_alpha)

    # ==========================================
    # CAKE OPTIMIZATION: Pre-build Static Topologies
    # ==========================================
    # Instead of rebuilding the graph every iteration, we build it once,
    # partition it so it stays on specific workers, and lock it in RAM.

    # Format: (user_id, [item_id_1, item_id_2, ...])
    user_topology = interactions_rdd.groupByKey().mapValues(list).partitionBy(num_partitions).cache()

    # Format: (item_id, [user_id_1, user_id_2, ...])
    item_topology = interactions_rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(list).partitionBy(num_partitions).cache()

    # ==========================================
    # INITIALIZATION (Co-partitioned with Topology)
    # ==========================================
    P_init = [(u, rng.normal(0, 0.01, k)) for u in range(M)]
    Q_init = [(i, rng.normal(0, 0.01, k)) for i in range(N)]

    # We partition P and Q identically to the topologies so joins require NO network shuffle!
    user_rdd = sc.parallelize(P_init).partitionBy(num_partitions).cache()

    raw_item_rdd = sc.parallelize(Q_init).partitionBy(num_partitions)
    item_rdd = raw_item_rdd.leftOuterJoin(item_weights_rdd) \
        .mapValues(lambda x: (x[0], x[1] if x[1] is not None else 0.0)) \
        .partitionBy(num_partitions).cache()

    losses = []
    if verbose:
        R_csr = train_matrix.tocsr()

    for iteration in range(num_iter):
        t_start = time.time()

        # ==========================================
        # ---- UPDATE P (USER FACTORS) ----
        # ==========================================
        Sq = item_rdd.map(lambda x: x[1][1] * np.outer(x[1][0], x[1][0])).sum()
        Sq_bc = sc.broadcast(Sq)

        # 1. Explode the static topology to fetch the current Item factors
        # Format: (item_id, user_id)
        user_requests = user_topology.flatMapValues(lambda items: items).map(lambda x: (x[1], x[0]))

        # 2. Join to get the current Q vectors: (item_id, (user_id, q_data))
        fetched_items = user_requests.join(item_rdd)

        # 3. Regroup back to users: (user_id, [(item_id, q_data), ...])
        # Because we partitionBy here, Spark routes the data directly to the co-located user partitions
        shopping_list_rdd = fetched_items.map(lambda x: (x[1][0], (x[0], x[1][1]))) \
                                         .groupByKey(num_partitions).mapValues(list)

        # 4. Local Join (Zero Shuffle because user_rdd is already partitioned the same way)
        user_update_data = user_rdd.leftOuterJoin(shopping_list_rdd)

        def update_user_partition(user_data):
            u, (p_u, item_list) = user_data
            p_u = p_u.copy()
            sq_local = Sq_bc.value

            if not item_list:
                return (u, p_u)

            Q_u = np.array([item[1][0] for item in item_list])
            C_u = np.array([item[1][1] for item in item_list])

            pred_cache = Q_u @ p_u

            for f in range(k):
                q_f = Q_u[:, f]
                hat_r = 1.0 - pred_cache + p_u[f] * q_f

                numer = np.dot(q_f, (c_obs - C_u) * hat_r)
                numer -= (p_u @ sq_local[:, f] - p_u[f] * sq_local[f, f])
                denom = np.dot(q_f ** 2, c_obs - C_u) + sq_local[f, f] + reg

                old_val = p_u[f]
                p_u[f] = numer / denom
                pred_cache += (p_u[f] - old_val) * q_f

            return (u, p_u)

        old_user_rdd = user_rdd
        # Ensure the new RDD maintains the partitioning scheme!
        user_rdd = user_update_data.map(update_user_partition).partitionBy(num_partitions)
        user_rdd.checkpoint()
        user_rdd.cache()
        user_rdd.count()

        old_user_rdd.unpersist()
        Sq_bc.unpersist()

        # ==========================================
        # ---- UPDATE Q (ITEM FACTORS) ----
        # ==========================================
        Sp = user_rdd.map(lambda x: np.outer(x[1], x[1])).sum()
        Sp_bc = sc.broadcast(Sp)

        # 1. Explode the static topology to fetch current User factors
        # Format: (user_id, item_id)
        item_requests = item_topology.flatMapValues(lambda users: users).map(lambda x: (x[1], x[0]))

        # 2. Join to get current P vectors
        fetched_users = item_requests.join(user_rdd)

        # 3. Regroup back to items
        item_shopping_list_rdd = fetched_users.map(lambda x: (x[1][0], (x[0], x[1][1]))) \
                                              .groupByKey(num_partitions).mapValues(list)

        # 4. Local Join
        item_update_data = item_rdd.leftOuterJoin(item_shopping_list_rdd)

        def update_item_partition(item_data):
            i, (q_data, user_list) = item_data
            q_i = q_data[0].copy()
            c_i = q_data[1]
            sp_local = Sp_bc.value

            if not user_list:
                return (i, (q_i, c_i))

            P_i = np.array([p_u for user_id, p_u in user_list])
            pred_cache = P_i @ q_i

            for f in range(k):
                p_f = P_i[:, f]
                hat_r = 1.0 - pred_cache + q_i[f] * p_f

                numer = np.dot(p_f, (c_obs - c_i) * hat_r)
                numer -= c_i * (q_i @ sp_local[:, f] - q_i[f] * sp_local[f, f])
                denom = (c_obs - c_i) * np.dot(p_f, p_f) + c_i * sp_local[f, f] + reg
                old_val = q_i[f]
                q_i[f] = numer / denom
                pred_cache += (q_i[f] - old_val) * p_f

            return (i, (q_i, c_i))

        old_item_rdd = item_rdd
        item_rdd = item_update_data.map(update_item_partition).partitionBy(num_partitions)
        item_rdd.checkpoint()
        item_rdd.cache()
        item_rdd.count()

        old_item_rdd.unpersist()
        Sp_bc.unpersist()

        elapsed = time.time() - t_start

        if verbose:
            P = np.zeros((M, k))
            for u, p_u in user_rdd.collect():
                P[u] = p_u

            Q = np.zeros((N, k))
            for i, (q_i, c_i) in item_rdd.collect():
                Q[i] = q_i

            loss = _compute_loss_fast(P, Q, R_csr, c_obs, c0, reg)
            losses.append(loss)
            print(f"Iteration {iteration + 1}/{num_iter} | loss={loss:.4f} | time={elapsed:.2f}s")

    # ==========================================
    # FINALIZATION
    # ==========================================
    P = np.zeros((M, k))
    for u, p_u in user_rdd.collect():
        P[u] = p_u

    Q = np.zeros((N, k))
    for i, (q_i, c_i) in item_rdd.collect():
        Q[i] = q_i

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
    sc.setCheckpointDir("/tmp/spark-checkpoints")

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
