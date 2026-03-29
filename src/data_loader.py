"""
Data Loading & Preprocessing for eALS with Implicit Feedback.

Handles:
- Loading MovieLens datasets (100K, 1M, 10M)
- Binarizing ratings into implicit feedback (interaction = 1, no interaction = 0)
- Re-indexing user/item IDs to contiguous 0-based indices
- Train/test splitting via leave-one-out (last interaction per user = test)
- Building sparse interaction matrices
- Subsampling for scalability experiments
"""

import os
import numpy as np
from scipy import sparse
from collections import defaultdict


def load_movielens_raw(data_dir, dataset="100k"):
    """
    Load raw MovieLens interactions as a list of (user, item, rating, timestamp).

    Args:
        data_dir: Path to the data/ directory containing ml-100k/, ml-1m/, etc.
        dataset: One of "100k", "1m", "10m".

    Returns:
        List of tuples (user_id_raw, item_id_raw, rating, timestamp).
    """
    interactions = []

    if dataset == "100k":
        filepath = os.path.join(data_dir, "ml-100k", "u.data")
        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                user, item, rating, ts = int(parts[0]), int(parts[1]), float(parts[2]), int(parts[3])
                interactions.append((user, item, rating, ts))

    elif dataset == "1m":
        filepath = os.path.join(data_dir, "ml-1m", "ratings.dat")
        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split("::")
                user, item, rating, ts = int(parts[0]), int(parts[1]), float(parts[2]), int(parts[3])
                interactions.append((user, item, rating, ts))

    elif dataset == "10m":
        filepath = os.path.join(data_dir, "ml-10M100K", "ratings.dat")
        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split("::")
                user, item, rating, ts = int(parts[0]), int(parts[1]), float(parts[2]), int(parts[3])
                interactions.append((user, item, rating, ts))
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use '100k', '1m', or '10m'.")

    print(f"Loaded {len(interactions)} raw interactions from MovieLens {dataset}")
    return interactions


def reindex_ids(interactions):
    """
    Re-map raw user/item IDs to contiguous 0-based indices.

    Returns:
        reindexed: List of (user_idx, item_idx, rating, timestamp)
        user_map: dict {raw_id -> 0-based index}
        item_map: dict {raw_id -> 0-based index}
        num_users: int
        num_items: int
    """
    user_ids = sorted(set(u for u, _, _, _ in interactions))
    item_ids = sorted(set(i for _, i, _, _ in interactions))

    user_map = {raw: idx for idx, raw in enumerate(user_ids)}
    item_map = {raw: idx for idx, raw in enumerate(item_ids)}

    reindexed = [
        (user_map[u], item_map[i], r, ts)
        for u, i, r, ts in interactions
    ]

    num_users = len(user_ids)
    num_items = len(item_ids)
    print(f"Re-indexed: {num_users} users, {num_items} items")

    return reindexed, user_map, item_map, num_users, num_items


def binarize(interactions, threshold=0.0):
    """
    Convert explicit ratings to implicit feedback.

    Any rating > threshold becomes 1 (observed interaction).
    For implicit feedback, we treat all observed interactions as positive.

    Args:
        interactions: List of (user, item, rating, timestamp)
        threshold: Minimum rating to count as positive (default 0 = all interactions)

    Returns:
        List of (user, item, 1.0, timestamp) for interactions above threshold.
    """
    binarized = [
        (u, i, 1.0, ts)
        for u, i, r, ts in interactions
        if r > threshold
    ]
    print(f"Binarized: {len(binarized)} positive interactions (threshold={threshold})")
    return binarized


def build_interaction_matrix(interactions, num_users, num_items):
    """
    Build a sparse user-item interaction matrix R (CSR format).

    R[u, i] = 1 if user u interacted with item i, else 0.

    Args:
        interactions: List of (user, item, value, timestamp)
        num_users, num_items: Matrix dimensions.

    Returns:
        R: scipy.sparse.csr_matrix of shape (num_users, num_items)
    """
    users = np.array([u for u, _, _, _ in interactions])
    items = np.array([i for _, i, _, _ in interactions])
    values = np.ones(len(interactions), dtype=np.float64)

    R = sparse.csr_matrix((values, (users, items)), shape=(num_users, num_items))
    # Deduplicate: clip values to 1 (in case of repeated interactions)
    R.data = np.clip(R.data, 0, 1)
    R.eliminate_zeros()

    density = R.nnz / (num_users * num_items) * 100
    print(f"Interaction matrix: {num_users}x{num_items}, "
          f"{R.nnz} non-zeros, density={density:.4f}%")

    return R


def leave_one_out_split(interactions, num_users, num_items):
    """
    Leave-one-out evaluation split (as used in the eALS paper).

    For each user, the most recent interaction goes to the test set;
    all other interactions form the training set.

    Args:
        interactions: List of (user, item, value, timestamp), sorted not required.
        num_users, num_items: Matrix dimensions.

    Returns:
        train_matrix: scipy.sparse.csr_matrix (num_users x num_items)
        test_dict: dict {user_id: item_id} — one test item per user
    """
    # Group interactions by user, sort by timestamp
    user_interactions = defaultdict(list)
    for u, i, v, ts in interactions:
        user_interactions[u].append((i, ts))

    train_rows, train_cols = [], []
    test_dict = {}

    for u in range(num_users):
        items_ts = user_interactions.get(u, [])
        if len(items_ts) == 0:
            continue

        # Sort by timestamp, last one is test
        items_ts.sort(key=lambda x: x[1])

        if len(items_ts) == 1:
            # Only one interaction: put in training (can't test this user)
            train_rows.append(u)
            train_cols.append(items_ts[0][0])
        else:
            # Last interaction = test
            test_dict[u] = items_ts[-1][0]
            # Rest = train
            for item_id, _ in items_ts[:-1]:
                train_rows.append(u)
                train_cols.append(item_id)

    train_values = np.ones(len(train_rows), dtype=np.float64)
    train_matrix = sparse.csr_matrix(
        (train_values, (train_rows, train_cols)),
        shape=(num_users, num_items)
    )
    train_matrix.data = np.clip(train_matrix.data, 0, 1)
    train_matrix.eliminate_zeros()

    print(f"Train/test split: {train_matrix.nnz} train interactions, "
          f"{len(test_dict)} test users")

    return train_matrix, test_dict


def subsample(interactions, fraction, seed=42):
    """
    Subsample interactions for scalability experiments.

    Keeps all interactions for a random subset of users.

    Args:
        interactions: List of (user, item, value, timestamp)
        fraction: Fraction of users to keep (0.0 to 1.0)
        seed: Random seed for reproducibility.

    Returns:
        Subsampled list of interactions.
    """
    rng = np.random.RandomState(seed)
    all_users = sorted(set(u for u, _, _, _ in interactions))
    num_keep = max(1, int(len(all_users) * fraction))
    kept_users = set(rng.choice(all_users, size=num_keep, replace=False))

    subsampled = [
        (u, i, v, ts) for u, i, v, ts in interactions
        if u in kept_users
    ]
    print(f"Subsampled: kept {num_keep}/{len(all_users)} users, "
          f"{len(subsampled)}/{len(interactions)} interactions")

    return subsampled


def load_and_prepare(data_dir, dataset="100k", rating_threshold=0.0, subsample_frac=1.0):
    """
    Full pipeline: load -> reindex -> binarize -> split -> build matrices.

    Args:
        data_dir: Path to data/ directory.
        dataset: "100k", "1m", or "10m".
        rating_threshold: Min rating to count as positive.
        subsample_frac: Fraction of users to keep (1.0 = all).

    Returns:
        dict with keys:
            train_matrix, test_dict, num_users, num_items,
            user_map, item_map, full_matrix
    """
    raw = load_movielens_raw(data_dir, dataset)
    reindexed, user_map, item_map, num_users, num_items = reindex_ids(raw)
    binarized = binarize(reindexed, threshold=rating_threshold)

    if subsample_frac < 1.0:
        binarized = subsample(binarized, subsample_frac)
        # Re-reindex after subsampling
        binarized, user_map_sub, item_map_sub, num_users, num_items = reindex_ids(
            [(u, i, v, ts) for u, i, v, ts in binarized]
        )
        # Compose maps
        inv_user = {v: k for k, v in user_map.items()}
        inv_item = {v: k for k, v in item_map.items()}
        user_map = {inv_user[k]: v for k, v in user_map_sub.items()}
        item_map = {inv_item[k]: v for k, v in item_map_sub.items()}

    full_matrix = build_interaction_matrix(binarized, num_users, num_items)
    train_matrix, test_dict = leave_one_out_split(binarized, num_users, num_items)

    return {
        "train_matrix": train_matrix,
        "test_dict": test_dict,
        "full_matrix": full_matrix,
        "num_users": num_users,
        "num_items": num_items,
        "user_map": user_map,
        "item_map": item_map,
    }


if __name__ == "__main__":
    # Quick sanity check
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")

    result = load_and_prepare(data_dir, dataset="100k")

    print(f"\n--- Summary ---")
    print(f"Users: {result['num_users']}")
    print(f"Items: {result['num_items']}")
    print(f"Train nnz: {result['train_matrix'].nnz}")
    print(f"Test users: {len(result['test_dict'])}")
    print(f"Full matrix density: "
          f"{result['full_matrix'].nnz / (result['num_users'] * result['num_items']) * 100:.4f}%")

    # Verify test items are not in training
    train = result["train_matrix"]
    errors = 0
    for u, i in result["test_dict"].items():
        if train[u, i] != 0:
            errors += 1
    print(f"Leakage check: {errors} test items found in train (should be 0)")
