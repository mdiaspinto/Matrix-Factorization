"""
Data Loading & Preprocessing for eALS with Implicit Feedback.

Handles:
- Loading the Yelp review dataset (yelp_academic_dataset_review.json)
- Binarizing ratings into implicit feedback (interaction = 1, no interaction = 0)
- Re-indexing user/item IDs to contiguous 0-based indices
- Train/test splitting via leave-one-out (last interaction per user = test)
- Building sparse interaction matrices
- Subsampling for scalability experiments
"""

import os
import json
import numpy as np
from scipy import sparse
from collections import defaultdict
from datetime import datetime


def load_yelp_raw(data_dir):
    """
    Load raw Yelp interactions from yelp_academic_dataset_review.json.

    The file is in JSON-lines format: one JSON object per line.
    Each object has: user_id (str), business_id (str), stars (float), date (str).

    We treat each review as one implicit interaction regardless of star rating.
    The date string (e.g. "2016-05-28 00:00:00") is converted to a Unix
    timestamp integer for consistent chronological sorting in leave-one-out splitting.

    Args:
        data_dir: Path to the data/ directory containing yelp/ subdirectory.

    Returns:
        List of tuples (user_id_raw, business_id_raw, stars, timestamp_int).
    """
    filepath = os.path.join(data_dir, "yelp", "yelp_academic_dataset_review.json")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Yelp review file not found at {filepath}\n"
            f"Run download_data.sh first to extract the dataset."
        )

    interactions = []
    date_fmt = "%Y-%m-%d %H:%M:%S"

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            user_id = obj["user_id"]
            business_id = obj["business_id"]
            stars = float(obj["stars"])
            ts = int(datetime.strptime(obj["date"], date_fmt).timestamp())
            interactions.append((user_id, business_id, stars, ts))

    print(f"Loaded {len(interactions)} raw interactions from Yelp")
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


def filter_min_interactions(interactions, min_user=5, min_item=0):
    """
    Remove users (and optionally items) with fewer than N interactions.

    Filtering is applied iteratively since removing items can push users
    below the threshold and vice versa.

    Args:
        interactions: List of (user, item, value, timestamp).
        min_user: Minimum interactions per user to keep (default 5).
        min_item: Minimum interactions per item to keep (default 0 = no filter).

    Returns:
        Filtered list of interactions.
    """
    prev_count = len(interactions) + 1

    while len(interactions) < prev_count:
        prev_count = len(interactions)

        if min_user > 0:
            user_counts = defaultdict(int)
            for u, i, v, ts in interactions:
                user_counts[u] += 1
            interactions = [
                (u, i, v, ts) for u, i, v, ts in interactions
                if user_counts[u] >= min_user
            ]

        if min_item > 0:
            item_counts = defaultdict(int)
            for u, i, v, ts in interactions:
                item_counts[i] += 1
            interactions = [
                (u, i, v, ts) for u, i, v, ts in interactions
                if item_counts[i] >= min_item
            ]

    remaining_users = len(set(u for u, _, _, _ in interactions))
    remaining_items = len(set(i for _, i, _, _ in interactions))
    print(f"Filtered: {remaining_users} users (min={min_user}), "
          f"{remaining_items} items (min={min_item}), "
          f"{len(interactions)} interactions remaining")

    return interactions


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
            test_item = items_ts[-1][0]
            test_dict[u] = test_item
            
            # Rest = train (strictly excluding the test item to prevent leakage)
            for item_id, _ in items_ts[:-1]:
                if item_id != test_item:
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


def load_and_prepare(data_dir, rating_threshold=0.0,
                     min_user_interactions=10, min_item_interactions=10,
                     subsample_frac=1.0):
    """
    Full pipeline: load -> reindex -> binarize -> filter -> split -> build matrices.

    Filtering defaults (min 10 interactions per user and item) match the
    paper's preprocessing exactly.

    Args:
        data_dir: Path to data/ directory.
        rating_threshold: Min rating to count as positive (default 0 = all reviews).
        min_user_interactions: Drop users with fewer interactions (default 10).
        min_item_interactions: Drop items with fewer interactions (default 10).
        subsample_frac: Fraction of users to keep (1.0 = all).

    Returns:
        dict with keys:
            train_matrix, test_dict, num_users, num_items,
            user_map, item_map, full_matrix
    """
    raw = load_yelp_raw(data_dir)
    reindexed, user_map, item_map, num_users, num_items = reindex_ids(raw)
    binarized = binarize(reindexed, threshold=rating_threshold)

    if min_user_interactions > 0 or min_item_interactions > 0:
        binarized = filter_min_interactions(
            binarized, min_user=min_user_interactions,
            min_item=min_item_interactions
        )
        # Re-reindex after filtering
        binarized, user_map_filt, item_map_filt, num_users, num_items = reindex_ids(
            binarized
        )
        inv_user = {v: k for k, v in user_map.items()}
        inv_item = {v: k for k, v in item_map.items()}
        user_map = {inv_user[k]: v for k, v in user_map_filt.items()}
        item_map = {inv_item[k]: v for k, v in item_map_filt.items()}

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

    result = load_and_prepare(data_dir)

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
