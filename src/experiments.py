#!/usr/bin/env python3
"""
experiments.py — Experimental analysis for the eALS project.

Runs four experiments and saves results (CSVs + PNG plots) to results/.

Experiments:
  --scalability  Per-iteration time vs |R| at subsampled fractions (PRIMARY)
  --spark-cores  Spark local[1] vs local[*] speedup
  --convergence  Loss + HR@K/NDCG@K over 50 iterations (K=128)
  --k-sweep      Recommendation quality and speed for K in {8,16,32,64,128}

Usage:
  python experiments.py --data-dir ../data --all
  python experiments.py --data-dir ../data --scalability --spark-cores
  python experiments.py --data-dir ../data --convergence
"""

import argparse
import os
import time
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive: works on servers/headless
import matplotlib.pyplot as plt

from pyspark import SparkContext, SparkConf

# Project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader  import load_and_prepare
from eals_rdd     import eals_train_rdd
from evaluation   import evaluate_multiple_k

# ── Setup ────────────────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Paper's optimal hyperparameters for Yelp (He et al., Figures 1a/1b)
YELP_PARAMS = dict(lam=0.01, w_obs=0.0, c0=512, alpha=0.4)


def make_sc(app_name, cores="*"):
    """Create a local SparkContext. Stop any existing one first."""
    conf = (SparkConf()
            .setAppName(app_name)
            .setMaster(f"local[{cores}]")
            .set("spark.driver.memory", "4g"))
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    return sc


def subsample_matrix(train_csr, fraction, seed=42):
    """
    Subsample a fraction of users from a CSR sparse matrix.
    Returns (submatrix, num_users, num_items).
    Preserves only items that have at least one interaction after subsampling.
    """
    M = train_csr.shape[0]
    rng    = np.random.RandomState(seed)
    n_keep = max(1, int(M * fraction))
    kept   = sorted(rng.choice(M, size=n_keep, replace=False))
    sub    = train_csr[kept, :]
    # Remove empty item columns
    item_sums = np.array(sub.sum(axis=0)).flatten()
    keep_cols = np.where(item_sums > 0)[0]
    sub       = sub[:, keep_cols]
    return sub, sub.shape[0], sub.shape[1]


# ── Experiment 1: Scalability (PRIMARY) ──────────────────────────────────────

def run_scalability(data_dir, fractions=(0.25, 0.5, 0.75, 1.0),
                    K=32, num_iter=10):
    """
    Measure per-iteration time as the number of training interactions grows.

    Uses K=32 for speed; num_iter=10 is enough to reach steady-state timing.
    The theoretical prediction is near-linear growth in |R|, since
    O(|R|*K + (M+N)*K^2) is dominated by |R|*K for large datasets.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Scalability")
    print(f"  K={K}, {num_iter} iterations, fractions={fractions}")
    print("="*60)

    data         = load_and_prepare(data_dir)
    full_train   = data["train_matrix"]
    test_dict    = data["test_dict"]

    sc   = make_sc("eALS-scalability")
    rows = []

    for frac in fractions:
        print(f"\n--- Fraction {frac:.0%} ---")
        if frac < 1.0:
            train_sub, n_users, n_items = subsample_matrix(full_train, frac)
        else:
            train_sub, n_users, n_items = full_train, *full_train.shape

        print(f"  Users: {n_users:,}  |  Items: {n_items:,}  "
              f"|  NNZ: {train_sub.nnz:,}")

        t0              = time.time()
        P, Q, losses, _ = eals_train_rdd(
            sc, train_sub, K=K, num_iter=num_iter,
            verbose=True, **YELP_PARAMS)
        total_time    = time.time() - t0
        avg_iter_time = total_time / num_iter

        # Evaluate only on full dataset fraction (others lack test_dict mapping)
        if frac == 1.0:
            metrics = evaluate_multiple_k(P, Q, full_train, test_dict,
                                          k_values=(10,))
            hr10 = round(metrics["HR@10"], 4)
        else:
            hr10 = float("nan")

        rows.append({
            "fraction":         f"{frac:.0%}",
            "num_users":        n_users,
            "num_items":        n_items,
            "train_nnz":        train_sub.nnz,
            "avg_iter_time_s":  round(avg_iter_time, 2),
            "HR@10":            hr10,
        })
        print(f"  Avg iter time: {avg_iter_time:.2f}s | HR@10: {hr10}")

    sc.stop()

    df       = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "scalability.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(df.to_string(index=False))

    # ── Plot: log-log iteration time vs |R| ──────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    nnz_vals  = df["train_nnz"].values.astype(float)
    time_vals = df["avg_iter_time_s"].values.astype(float)

    ax.plot(nnz_vals, time_vals, "o-", color="#2E75B6",
            linewidth=2, markersize=8, label="eALS (Spark)")

    # Linear reference anchored at smallest point
    y_lin = time_vals[0] * (nnz_vals / nnz_vals[0])
    ax.plot(nnz_vals, y_lin, "--", color="gray",
            alpha=0.7, linewidth=1.5, label="Linear reference")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Training interactions $|R|$", fontsize=12)
    ax.set_ylabel("Avg. time per iteration (s)", fontsize=12)
    ax.set_title(f"Scalability  (K={K}, {num_iter} iterations)", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "scalability.png"), dpi=150)
    plt.close()
    print("Plot saved: results/scalability.png")
    return df


# ── Experiment 2: Spark Cores Speedup ────────────────────────────────────────

def run_spark_cores(data_dir, K=32, num_iter=10):
    """
    Compare Spark local[1] (sequential) vs local[*] (all cores).
    Demonstrates the parallel speedup from the embarrassingly-parallel
    user/item update steps.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Spark Cores Speedup")
    print(f"  K={K}, {num_iter} iterations, local[1] vs local[*]")
    print("="*60)

    data         = load_and_prepare(data_dir)
    train_matrix = data["train_matrix"]
    test_dict    = data["test_dict"]

    rows = []

    for cores in [1, "*"]:
        label = f"local[{cores}]"
        print(f"\n--- {label} ---")

        sc              = make_sc(f"eALS-cores", cores=cores)
        t0              = time.time()
        P, Q, losses, _ = eals_train_rdd(
            sc, train_matrix, K=K, num_iter=num_iter,
            verbose=True, **YELP_PARAMS)
        total_time = time.time() - t0
        sc.stop()

        metrics = evaluate_multiple_k(P, Q, train_matrix, test_dict,
                                      k_values=(10,))
        rows.append({
            "configuration":    label,
            "total_time_s":     round(total_time, 2),
            "avg_iter_time_s":  round(total_time / num_iter, 2),
            "HR@10":            round(metrics["HR@10"], 4),
            "NDCG@10":          round(metrics["NDCG@10"], 4),
        })
        print(f"  Total: {total_time:.2f}s | HR@10: {metrics['HR@10']:.4f}")

    df        = pd.DataFrame(rows)
    t_single  = df.loc[df["configuration"] == "local[1]",
                       "total_time_s"].iloc[0]
    df["speedup"] = (t_single / df["total_time_s"]).round(2)

    csv_path = os.path.join(RESULTS_DIR, "spark_cores.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(df.to_string(index=False))

    # ── Plot: bar chart ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    colours = ["#BFBFBF", "#2E75B6"]
    bars    = ax.bar(df["configuration"], df["total_time_s"],
                     color=colours, width=0.5, edgecolor="white")
    ymax = df["total_time_s"].max()
    for bar, row in zip(bars, df.itertuples()):
        if row.configuration != "local[1]":
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ymax * 0.02,
                    f"{row.speedup}\u00d7",
                    ha="center", va="bottom",
                    fontsize=13, fontweight="bold", color="#2E75B6")
    ax.set_ylabel("Total training time (s)", fontsize=12)
    ax.set_title(f"Spark Speedup  (K={K}, {num_iter} iterations)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "spark_cores.png"), dpi=150)
    plt.close()
    print("Plot saved: results/spark_cores.png")
    return df


# ── Experiment 3: Convergence ─────────────────────────────────────────────────

def run_convergence(data_dir, K=128, num_iter=50):
    """
    Track training loss and HR@K / NDCG@K after every iteration.
    Uses paper hyperparameters (K=128, c0=512, alpha=0.4) for Yelp.
    eval_fn is passed to eals_train_rdd to collect per-iteration metrics.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Convergence")
    print(f"  K={K}, {num_iter} iterations — evaluating every iteration")
    print("="*60)

    data         = load_and_prepare(data_dir)
    train_matrix = data["train_matrix"]
    test_dict    = data["test_dict"]

    sc = make_sc("eALS-convergence")

    def eval_fn(P, Q):
        """Called by eals_train_rdd after each iteration."""
        return evaluate_multiple_k(P, Q, train_matrix, test_dict,
                                   k_values=(5, 10, 20))

    print("\nTraining with per-iteration evaluation (this is the slow path)...\n")
    P, Q, losses, eval_history = eals_train_rdd(
        sc, train_matrix,
        K=K, num_iter=num_iter,
        verbose=True,
        eval_fn=eval_fn,
        **YELP_PARAMS)
    sc.stop()

    df       = pd.DataFrame(eval_history)
    csv_path = os.path.join(RESULTS_DIR, "convergence.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(df.to_string(index=False))

    # ── Plot: loss + HR + NDCG ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    iters     = df["iteration"]
    colours   = {"5": "#ED7D31", "10": "#2E75B6", "20": "#70AD47"}

    axes[0].plot(iters, df["loss"], color="#C00000", linewidth=2)
    axes[0].set_title("Training Loss", fontsize=12)
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Weighted Squared Loss")
    axes[0].grid(alpha=0.3)

    for k_val in [5, 10, 20]:
        col = colours[str(k_val)]
        axes[1].plot(iters, df[f"HR@{k_val}"],
                     label=f"HR@{k_val}", linewidth=2, color=col)
    axes[1].set_title("Hit Rate", fontsize=12)
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("HR@K")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    for k_val in [5, 10, 20]:
        col = colours[str(k_val)]
        axes[2].plot(iters, df[f"NDCG@{k_val}"],
                     label=f"NDCG@{k_val}", linewidth=2, color=col)
    axes[2].set_title("NDCG", fontsize=12)
    axes[2].set_xlabel("Iteration"); axes[2].set_ylabel("NDCG@K")
    axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

    plt.suptitle(f"Convergence — Yelp (K={K})", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "convergence.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Plot saved: results/convergence.png")
    return df


# ── Experiment 4: K Sweep ─────────────────────────────────────────────────────

def run_k_sweep(data_dir, k_values=(8, 16, 32, 64, 128), num_iter=20):
    """
    Vary K and measure recommendation quality and per-iteration time.
    Uses num_iter=20 (enough to converge for all K).
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: K Sweep")
    print(f"  K in {k_values}, {num_iter} iterations each")
    print("="*60)

    data         = load_and_prepare(data_dir)
    train_matrix = data["train_matrix"]
    test_dict    = data["test_dict"]

    sc   = make_sc("eALS-ksweep")
    rows = []

    for K in k_values:
        print(f"\n--- K={K} ---")
        t0              = time.time()
        P, Q, losses, _ = eals_train_rdd(
            sc, train_matrix,
            K=K, num_iter=num_iter,
            verbose=False, **YELP_PARAMS)
        total_time = time.time() - t0
        avg_iter   = total_time / num_iter

        metrics = evaluate_multiple_k(P, Q, train_matrix, test_dict,
                                      k_values=(5, 10, 20))
        rows.append({
            "K":               K,
            "HR@5":            round(metrics["HR@5"],   4),
            "HR@10":           round(metrics["HR@10"],  4),
            "HR@20":           round(metrics["HR@20"],  4),
            "NDCG@10":         round(metrics["NDCG@10"], 4),
            "avg_iter_time_s": round(avg_iter, 2),
        })
        print(f"  HR@10={metrics['HR@10']:.4f} | "
              f"NDCG@10={metrics['NDCG@10']:.4f} | "
              f"avg iter={avg_iter:.2f}s")

    sc.stop()

    df       = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "k_sweep.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(df.to_string(index=False))

    # ── Plot: quality (left) + time (right) vs K ─────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(df["K"], df["HR@10"],   "o-", color="#2E75B6",
             linewidth=2, markersize=7, label="HR@10")
    ax1.plot(df["K"], df["NDCG@10"], "s--", color="#ED7D31",
             linewidth=2, markersize=7, label="NDCG@10")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(list(k_values))
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.set_xlabel("K (latent factors)", fontsize=12)
    ax1.set_ylabel("Metric value", fontsize=12)
    ax1.set_title("Recommendation Quality vs K", fontsize=13)
    ax1.legend(fontsize=10); ax1.grid(alpha=0.3)

    ax2.bar([str(k) for k in df["K"]], df["avg_iter_time_s"],
            color="#2E75B6", alpha=0.85, edgecolor="white")
    ax2.set_xlabel("K (latent factors)", fontsize=12)
    ax2.set_ylabel("Avg. iter time (s)", fontsize=12)
    ax2.set_title("Computation Time vs K", fontsize=13)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "k_sweep.png"), dpi=150)
    plt.close()
    print("Plot saved: results/k_sweep.png")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run eALS experimental analysis on the Yelp dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--data-dir",    default="../data",
                        help="Path to data/ directory (default: ../data)")
    parser.add_argument("--all",         action="store_true",
                        help="Run all four experiments")
    parser.add_argument("--scalability", action="store_true",
                        help="Per-iteration time vs |R| (PRIMARY)")
    parser.add_argument("--spark-cores", action="store_true",
                        help="local[1] vs local[*] speedup")
    parser.add_argument("--convergence", action="store_true",
                        help="Loss + HR@K/NDCG@K over iterations")
    parser.add_argument("--k-sweep",     action="store_true",
                        help="Quality and speed for K in {8,16,32,64,128}")
    args = parser.parse_args()

    run_any = any([args.all, args.scalability, args.spark_cores,
                   args.convergence, args.k_sweep])
    if not run_any:
        parser.print_help()
        return

    print(f"\nResults will be saved to: {RESULTS_DIR}")
    print(f"Data directory:            {args.data_dir}\n")

    all_results = {}

    # Primary experiment: run first
    if args.all or args.scalability:
        all_results["scalability"] = run_scalability(args.data_dir)

    if args.all or args.spark_cores:
        all_results["spark_cores"] = run_spark_cores(args.data_dir)

    if args.all or args.convergence:
        all_results["convergence"] = run_convergence(args.data_dir)

    if args.all or args.k_sweep:
        all_results["k_sweep"] = run_k_sweep(args.data_dir)

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}/")
    print("="*60)
    for name, df in all_results.items():
        print(f"\n{name.upper()}:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
