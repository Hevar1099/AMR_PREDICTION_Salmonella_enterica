import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (
    classification_report,
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

STABILITY_ITERATIONS = 100
SUBSAMPLE_FRACTION = 0.5
SELECTION_THRESHOLD = 0.1
UNIVARIATE_FILTER_K = 50000
N_JOBS = -1

os.makedirs("Output", exist_ok=True)


# ==========================================
# Stability Selection for K-mers
# ==========================================
def _fit_kmer_subsample(seed, X, y, fraction):
    n_samples = X.shape[0]
    rng = np.random.RandomState(seed)
    indices = rng.choice(n_samples, int(n_samples * fraction), replace=False)
    X_sub = X[indices]
    y_sub = y[indices]

    if len(np.unique(y_sub)) < 2:
        return None
    clf = LogisticRegression(
        penalty="l1",
        C=0.1,
        solver="liblinear",
        class_weight="balanced",
        random_state=seed,
        max_iter=1000,
    )
    clf.fit(X_sub, y_sub)
    return (np.abs(clf.coef_) > 1e-5).flatten()


# ==========================================
# Gene Model Stability Audit
# ==========================================
def get_gene_stability():
    if not os.path.exists(GENE_DATA):
        print(f"{GENE_DATA} not found. Skipping gene audit.")
        return np.array([])
    df_gene = pd.read_csv(GENE_DATA)
    y_g = df_gene["label"]
    X_g = df_gene.drop(columns=["label"], errors="ignore")
    X_g = X_g.select_dtypes(include=[np.number])

    print(f"Auditing Gene Model Stability ({X_g.shape[1]} numeric features)...")

    def run_one(seed):

        rng = np.random.RandomState(seed)

        idx = rng.choice(len(y_g), int(len(y_g) * 0.5), replace=False)

        model = LogisticRegression(
            C=1.0,
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            random_state=seed,
            max_iter=1000,
        )

        model.fit(X_g.iloc[idx], y_g.iloc[idx])

        return (np.abs(model.coef_) > 1e-5).flatten()

    res = Parallel(n_jobs=N_JOBS)(
        delayed(run_one)(i) for i in range(STABILITY_ITERATIONS)
    )

    res = [r for r in res if r is not None]

    if len(res) == 0:
        return np.zeros(X_g.shape[1])

    return np.mean(res, axis=0)


# ==========================================
# Plot Stability Decay
# ==========================================


def plot_stability_comparison(kmer_scores_full, gene_scores):

    print("Generating stability decay plot...")

    thresholds = np.linspace(0, 1.0, 100)

    kmer_counts = [np.sum(kmer_scores_full >= t) for t in thresholds]
    gene_counts = [np.sum(gene_scores >= t) for t in thresholds]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, kmer_counts, color="darkred", lw=2.5, label="K-mer Model")
    plt.plot(thresholds, gene_counts, color="blue", lw=2.5, label="Gene Model")
    plt.yscale("log")
    plt.title("Feature Stability Decay", fontweight="bold")
    plt.xlabel("Stability Threshold")
    plt.ylabel("Stable Features (log scale)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FAILURE_PLOT, dpi=300)


# ==========================================
# Main Pipeline
# ==========================================
def main():

    print("Loading Sparse K-mer Matrix...")
    X_k = sparse.load_npz(KMER_MATRIX)
    y_k = pd.read_csv(LABEL_FILE)["label"].values
    print(f"Loaded: {X_k.shape[0]} samples, {X_k.shape[1]} features.")
    print(f"Tier 1: Filtering to Top {UNIVARIATE_FILTER_K} k-mers via Chi2...")

    selector = SelectKBest(chi2, k=min(UNIVARIATE_FILTER_K, X_k.shape[1]))
    X_k_reduced = selector.fit_transform(X_k, y_k)
    selected_indices = selector.get_support(indices=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_k_reduced, y_k, test_size=0.2, stratify=y_k, random_state=42
    )
    # ---------------------------------------------------------------------------
    # Tier 2: Stability Selection
    # ---------------------------------------------------------------------------
    print(f"Tier 2: Stability Selection ({STABILITY_ITERATIONS} iterations)...")

    kmer_results = Parallel(n_jobs=N_JOBS)(
        delayed(_fit_kmer_subsample)(i, X_train, y_train, SUBSAMPLE_FRACTION)
        for i in range(STABILITY_ITERATIONS)
    )
    k_scores_reduced = np.zeros(X_k_reduced.shape[1])
    valid_runs = 0
    for res in kmer_results:
        if res is not None:
            k_scores_reduced += res.astype(int)
            valid_runs += 1

    if valid_runs == 0:
        raise ValueError("No valid stability runs.")
    k_scores_reduced /= valid_runs

    # Map back to full space
    kmer_scores_full = np.zeros(X_k.shape[1])
    kmer_scores_full[selected_indices] = k_scores_reduced
    # ---------------------------------------------------------------------------
    # Gene Stability Comparison
    # ---------------------------------------------------------------------------
    gene_scores = get_gene_stability()
    if len(gene_scores) > 0:
        plot_stability_comparison(kmer_scores_full, gene_scores)
    # ---------------------------------------------------------------------------
    # Final Model
    # ---------------------------------------------------------------------------
    stable_mask = k_scores_reduced >= SELECTION_THRESHOLD
    n_selected = np.sum(stable_mask)
    print(f"Selected {n_selected} stable k-mers.")
    if n_selected == 0:
        print("Relaxing stability threshold...")
        stable_mask = k_scores_reduced >= 0.01
        n_selected = np.sum(stable_mask)
    if n_selected > 0:
        X_train_final = X_train[:, stable_mask]
        X_test_final = X_test[:, stable_mask]
        final_clf = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            l1_ratio=0.5,
            alpha=0.001,
            class_weight="balanced",
            random_state=42,
        )
        final_clf.fit(X_train_final, y_train)
        y_pred = final_clf.predict(X_test_final)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        vme = (fn / (fn + tp)) * 100 if (fn + tp) > 0 else 0
        print("\n--- Final Performance ---")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
        print(f"Very Major Error (VME): {vme:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    else:
        print("No predictive features identified.")


if __name__ == "__main__":
    main()
