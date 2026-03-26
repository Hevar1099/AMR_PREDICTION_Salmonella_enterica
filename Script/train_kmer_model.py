import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from joblib import Parallel, delayed
import os

# ==========================================
# Configuration
# ==========================================
KMER_MATRIX = "Data/X_kmer_matrix.npz"
LABEL_FILE = "Data/y_kmer_labels.csv"
GENE_DATA = "Data/final_ml_dataset.csv"
OUTPUT_METRICS = "Output/kmer_model_metrics.csv"
OUTPUT_PLOT = "Output/kmer_confusion_matrix.png"
FAILURE_PLOT = "Output/Figure9_Stability_Decay.png"
STABILITY_ITERATIONS = 50
SUBSAMPLE_FRACTION = 0.5
SELECTION_THRESHOLD = 0.5
N_JOBS = -1

os.makedirs("Output", exist_ok=True)

# ==========================================
# Part 1: Stability Functions
# ==========================================


def _fit_kmer_subsample(seed, X, y, fraction):
    n_samples = X.shape[0]
    rng = np.random.RandomState(seed)
    indices = rng.choice(n_samples, int(n_samples * fraction), replace=False)
    X_sub, y_sub = X[indices], y[indices]
    if len(np.unique(y_sub)) < 2:
        return None
    clf = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        l1_ratio=0.7,
        alpha=0.001,
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
    )
    clf.fit(X_sub, y_sub)
    return np.abs(clf.coef_[0]) > 1e-6


def get_gene_stability():
    """Calculates actual stability for the filtered gene dataset."""
    df_gene = pd.read_csv(GENE_DATA)
    X_g = df_gene.drop(columns=["label", "Assembly", "Serovar"])
    y_g = df_gene["label"]

    def run_one(seed):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(y_g), int(len(y_g) * 0.5), replace=False)
        model = LogisticRegression(
            C=1,
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            random_state=seed,
        )
        model.fit(X_g.iloc[idx], y_g.iloc[idx])
        return np.abs(model.coef_[0]) > 1e-5

    print(f"Auditing Gene Model Stability ({len(X_g.columns)} features)...")
    res = Parallel(n_jobs=N_JOBS)(
        delayed(run_one)(i) for i in range(STABILITY_ITERATIONS)
    )
    return np.mean(res, axis=0)


# ==========================================
# Part 2: Plotting Logic
# ==========================================


def plot_stability_comparison(kmer_scores, gene_scores):
    print("Generating Figure 9: Stability Decay Comparison...")
    thresholds = np.linspace(0, 1.0, 100)
    kmer_counts = [np.sum(kmer_scores >= t) for t in thresholds]
    gene_counts = [np.sum(gene_scores >= t) for t in thresholds]

    plt.figure(figsize=(10, 6))
    plt.plot(
        thresholds,
        kmer_counts,
        color="darkred",
        linewidth=2.5,
        label="K-mer Model (Agnostic)",
    )
    plt.plot(
        thresholds,
        gene_counts,
        color="blue",
        linewidth=2.5,
        label="Gene-Based Model (Knowledge-Filtered)",
    )

    plt.yscale("log")
    plt.title(
        "Figure 9: Feature Stability Decay (K-mer vs. Gene)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Stability Threshold (Fraction of Runs)")
    plt.ylabel("Stable Features Remaining (Log Scale)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()

    final_stable_genes = np.sum(gene_scores >= 0.5)
    plt.annotate(
        f"Stable Genes: {final_stable_genes}",
        xy=(0.5, final_stable_genes),
        xytext=(0.6, final_stable_genes * 10),
        arrowprops=dict(facecolor="blue", shrink=0.05),
        color="blue",
    )

    plt.tight_layout()
    plt.savefig(FAILURE_PLOT, dpi=300)
    print(f"Figure 9 saved to {FAILURE_PLOT}")


# ==========================================
# Part 3: Main Pipeline
# ==========================================


def main():
    X_k, y_k = sparse.load_npz(KMER_MATRIX), pd.read_csv(LABEL_FILE)["label"].values
    print(f"K-mer Data Loaded: {X_k.shape[0]} x {X_k.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_k, y_k, test_size=0.2, stratify=y_k, random_state=42
    )

    # Run K-mer Stability Selection
    print(f"Running K-mer Stability Selection ({STABILITY_ITERATIONS} iterations)...")
    kmer_results = Parallel(n_jobs=N_JOBS)(
        delayed(_fit_kmer_subsample)(i, X_train, y_train, SUBSAMPLE_FRACTION)
        for i in range(STABILITY_ITERATIONS)
    )
    kmer_scores = np.zeros(X_k.shape[1])
    valid_runs = 0
    for res in kmer_results:
        if res is not None:
            kmer_scores += res.astype(int)
            valid_runs += 1
    kmer_scores /= valid_runs

    gene_scores = get_gene_stability()

    # Generate Comparative Plot
    plot_stability_comparison(kmer_scores, gene_scores)

    n_selected = np.sum(kmer_scores >= SELECTION_THRESHOLD)
    print(f"Selected {n_selected} stable k-mers.")

    if n_selected == 0:
        print("\nCRITICAL: Zero K-mers passed stability threshold.")
        return

    X_train_reduced = X_train[:, kmer_scores >= SELECTION_THRESHOLD]
    X_test_reduced = X_test[:, kmer_scores >= SELECTION_THRESHOLD]
    final_clf = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        l1_ratio=0.5,
        alpha=0.0001,
        class_weight="balanced",
        random_state=42,
    )
    final_clf.fit(X_train_reduced, y_train)

    y_pred = final_clf.predict(X_test_reduced)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    vme = (fn / (fn + tp)) * 100 if (fn + tp) > 0 else 0
    print(f"K-mer Model VME: {vme:.2f}%")


if __name__ == "__main__":
    main()
