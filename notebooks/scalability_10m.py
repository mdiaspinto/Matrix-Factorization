"""
Broad scalability experiment across MovieLens 100K, 1M, and 10M.

For 10M, we subsample users since the pure-Python eALS loop is too slow
for the full 70K-user dataset on a single machine. This is itself a finding:
it demonstrates exactly where distributed computing becomes necessary.
"""

import sys, os, time, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import load_and_prepare
from eals_numpy import eals_train as eals_numpy_train
from evaluation import evaluate_multiple_k

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 42
K = 32
REG = 0.01
ALPHA = 1.0
C0 = 1.0
NUM_ITER = 10

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 12,
})


# =================================================================
# Part 1: Cross-dataset scalability (NumPy)
# =================================================================

print("=" * 70)
print("PART 1: Cross-dataset scalability (NumPy, k=32, 10 iters)")
print("=" * 70)

# For 10M we use 10% subsample (~7K users, ~1M interactions)
# This gives us a data point between 1M-full and 10M-full
RUNS = [
    ("100k", 1.0),
    ("1m",   1.0),
    ("10m",  0.05),
    ("10m",  0.10),
    ("10m",  0.25),
]

cross_results = []

for dataset_name, frac in RUNS:
    label = f"{dataset_name}" if frac == 1.0 else f"{dataset_name}@{int(frac*100)}%"
    print(f"\n--- {label} ---")

    t_load = time.time()
    data = load_and_prepare(
        DATA_DIR, dataset=dataset_name,
        min_user_interactions=5, subsample_frac=frac
    )
    load_time = time.time() - t_load

    M, N = data["num_users"], data["num_items"]
    nnz = data["train_matrix"].nnz
    density = nnz / (M * N) * 100
    print(f"  Users={M}, Items={N}, NNZ={nnz}, Density={density:.4f}%")
    print(f"  Load time: {load_time:.2f}s")

    t_train = time.time()
    P, Q, losses = eals_numpy_train(
        data["train_matrix"], num_factors=K, num_iter=NUM_ITER,
        reg=REG, alpha=ALPHA, c0=C0, seed=SEED, verbose=True
    )
    total_train = time.time() - t_train
    avg_iter = total_train / NUM_ITER

    metrics = evaluate_multiple_k(P, Q, data["train_matrix"], data["test_dict"], k_values=(5, 10, 20))

    row = {
        "label": label,
        "dataset": dataset_name,
        "fraction": frac,
        "num_users": M,
        "num_items": N,
        "train_nnz": nnz,
        "density_pct": round(density, 4),
        "load_time_s": round(load_time, 2),
        "total_train_s": round(total_train, 2),
        "avg_iter_s": round(avg_iter, 2),
        "final_loss": round(losses[-1], 1),
    }
    row.update(metrics)
    cross_results.append(row)

    print(f"  Train: {total_train:.1f}s total, {avg_iter:.2f}s/iter")
    print(f"  HR@10={metrics['HR@10']:.4f}, NDCG@10={metrics['NDCG@10']:.4f}")

cross_df = pd.DataFrame(cross_results)

print("\n\nCross-dataset summary:")
print(cross_df[["label", "num_users", "num_items", "train_nnz",
                "avg_iter_s", "HR@10", "NDCG@10"]].to_string(index=False))


# =================================================================
# Part 2: NumPy vs Spark RDD on 100K and 1M
# =================================================================

print("\n" + "=" * 70)
print("PART 2: NumPy vs Spark RDD (k=32, 10 iters)")
print("=" * 70)

from pyspark import SparkContext, SparkConf
from eals_rdd import eals_train_rdd

os.environ["JAVA_HOME"] = "/Users/mdiaspinto/miniforge3/lib/jvm"

conf = SparkConf().setAppName("eALS-Scalability").setMaster("local[*]")
conf.set("spark.driver.memory", "4g")
sc = SparkContext.getOrCreate(conf=conf)
sc.setLogLevel("ERROR")
print(f"Spark: {sc.master}, {sc.defaultParallelism} cores")

impl_rows = []

for ds_name in ["100k", "1m"]:
    print(f"\n--- MovieLens {ds_name} ---")
    data_ds = load_and_prepare(DATA_DIR, dataset=ds_name, min_user_interactions=5)

    # NumPy
    print(f"  NumPy...")
    t0 = time.time()
    P_np, Q_np, losses_np = eals_numpy_train(
        data_ds["train_matrix"], num_factors=K, num_iter=NUM_ITER,
        reg=REG, alpha=ALPHA, c0=C0, seed=SEED, verbose=False
    )
    time_np = time.time() - t0
    metrics_np = evaluate_multiple_k(P_np, Q_np, data_ds["train_matrix"], data_ds["test_dict"], k_values=(10,))

    # Spark RDD
    print(f"  Spark RDD...")
    t0 = time.time()
    P_rdd, Q_rdd, losses_rdd = eals_train_rdd(
        sc, data_ds["train_matrix"], num_factors=K, num_iter=NUM_ITER,
        reg=REG, alpha=ALPHA, c0=C0, seed=SEED, verbose=False
    )
    time_rdd = time.time() - t0
    metrics_rdd = evaluate_multiple_k(P_rdd, Q_rdd, data_ds["train_matrix"], data_ds["test_dict"], k_values=(10,))

    match = np.allclose(losses_np, losses_rdd, rtol=1e-6)
    speedup = time_np / time_rdd

    impl_rows.append({
        "Dataset": f"ML-{ds_name}",
        "Users": data_ds["num_users"],
        "Train NNZ": data_ds["train_matrix"].nnz,
        "NumPy Total (s)": round(time_np, 2),
        "Spark Total (s)": round(time_rdd, 2),
        "Speedup": round(speedup, 2),
        "Loss Match": match,
        "HR@10 (NumPy)": metrics_np["HR@10"],
        "HR@10 (Spark)": metrics_rdd["HR@10"],
    })
    print(f"  NumPy={time_np:.2f}s, Spark={time_rdd:.2f}s, Speedup={speedup:.1f}x, Match={match}")

sc.stop()
print("Spark stopped.")

impl_df = pd.DataFrame(impl_rows)
print("\n")
print(impl_df.to_string(index=False))


# =================================================================
# Part 3: Generate figures
# =================================================================

print("\n" + "=" * 70)
print("Generating figures...")
print("=" * 70)

# Figure 1: Cross-dataset time scaling (log-log)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(cross_df["train_nnz"], cross_df["avg_iter_s"], "b-o", markersize=8, linewidth=2)
for _, row in cross_df.iterrows():
    axes[0].annotate(row["label"], (row["train_nnz"], row["avg_iter_s"]),
                     textcoords="offset points", xytext=(8, 4), fontsize=9)
# Linear reference
x = cross_df["train_nnz"].values.astype(float)
y0, x0 = cross_df["avg_iter_s"].values[0], float(x[0])
axes[0].plot(x, y0 * (x / x0), "k--", alpha=0.4, label="Linear reference")
axes[0].set_xlabel("Training Interactions")
axes[0].set_ylabel("Avg. Time per Iteration (s)")
axes[0].set_title("Per-Iteration Cost Across Datasets")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].legend()

# Quality across datasets
axes[1].plot(cross_df["train_nnz"], cross_df["HR@10"], "b-s", markersize=8, label="HR@10")
axes[1].plot(cross_df["train_nnz"], cross_df["NDCG@10"], "r-^", markersize=8, label="NDCG@10")
for _, row in cross_df.iterrows():
    axes[1].annotate(row["label"], (row["train_nnz"], row["HR@10"]),
                     textcoords="offset points", xytext=(8, 4), fontsize=9)
axes[1].set_xlabel("Training Interactions")
axes[1].set_ylabel("Metric Value")
axes[1].set_title("Recommendation Quality vs. Scale")
axes[1].set_xscale("log")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cross_dataset_scalability.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved cross_dataset_scalability.png")

# Figure 2: NumPy vs Spark bar chart
fig, ax = plt.subplots(figsize=(8, 5))
x_pos = range(len(impl_df))
width = 0.35
ax.bar([p - width/2 for p in x_pos], impl_df["NumPy Total (s)"], width,
       label="NumPy", color="steelblue")
ax.bar([p + width/2 for p in x_pos], impl_df["Spark Total (s)"], width,
       label="Spark RDD", color="coral")
ax.set_xticks(list(x_pos))
ax.set_xticklabels(impl_df["Dataset"])
ax.set_ylabel(f"Total Training Time (s) — {NUM_ITER} iterations")
ax.set_title(f"NumPy vs. Spark RDD (k={K})")
ax.legend()

for i, row in impl_df.iterrows():
    y_max = max(row["NumPy Total (s)"], row["Spark Total (s)"])
    ax.annotate(f"{row['Speedup']}x",
                xy=(i, y_max), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "numpy_vs_spark_datasets.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved numpy_vs_spark_datasets.png")

# Save all results
results = {
    "cross_dataset": cross_df.to_dict(orient="records"),
    "impl_comparison": impl_df.to_dict(orient="records"),
}
with open(os.path.join(FIG_DIR, "scalability_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)
print("  Saved scalability_results.json")

print("\nDone.")
