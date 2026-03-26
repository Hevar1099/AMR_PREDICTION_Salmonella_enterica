import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.spatial.distance import cdist
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from cycler import cycler


okabe_ito_palette = [
    "#56B4E9",
    "#D55E00",
    "#009E73",
    "#E69F00",
    "#CC79A7",
    "#0072B2",
    "#F0E442",
]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "axes.grid": True,
        "grid.color": "lightgray",
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "axes.prop_cycle": cycler("color", okabe_ito_palette),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    }
)

DATA_PATH = "Data/final_ml_dataset.csv"
OUTPUT_PATH = "Output/loso_results.csv"
PLOT_PATH = "Output/loso_performance.png"

os.makedirs("Output", exist_ok=True)

print("Loading data...")
df = pd.read_csv(DATA_PATH)


cols_to_drop = [
    "label",
    "Assembly",
    "Serovar",
    "#BioSample",
    "AMR genotypes",
    "AST phenotypes",
    "cipro_status",
    "Serovar_Clean",
]
existing_drop_cols = [c for c in cols_to_drop if c in df.columns]

X = df.drop(columns=existing_drop_cols).select_dtypes(include=[np.number])
y = df["label"]
groups = df["Serovar"]


scale_pos_weight = (y == 0).sum() / (y == 1).sum()
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
)

logo = LeaveOneGroupOut()
results = []

print(f"Running LOSO Cross-Validation across {groups.nunique()} serovars...\n")
print("-" * 120)
print(
    f"{'Serovar':<20} | {'AUC':<8} | {'F1':<8} | {'VME% (FN)':<10} | {'ME% (FP)':<10}"
)
print("-" * 120)

for train_idx, test_idx in logo.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    test_serovar = groups.iloc[test_idx].iloc[0]
    n_test_samples = len(test_idx)
    n_resistant = y_test.sum()

    distances = cdist(X_test, X_train, metric="jaccard")
    min_distances = np.min(distances, axis=1)
    mean_genetic_distance = np.mean(min_distances)
    # Skip AUC calculation if the serovar is 100% one class
    if len(np.unique(y_test)) < 2:
        print(
            f"Skipped {test_serovar}: Contains only 1 class (Resistant={n_resistant}/{n_test_samples})"
        )
        continue

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Standard Metrics
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    # Clinical Error Metrics (VME and ME)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    vme = (fn / (fn + tp)) * 100 if (fn + tp) > 0 else 0
    me = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0

    results.append(
        {
            "Left_Out_Serovar": test_serovar,
            "Test_Size": n_test_samples,
            "Resistant_Count": n_resistant,
            "Accuracy": acc,
            "ROC_AUC": auc,
            "F1_Score": f1,
            "VME_Rate": vme,
            "ME_Rate": me,
            "Genetic_Distance": mean_genetic_distance,
        }
    )

    print(f"{test_serovar:<20} | {auc:.4f}   | {f1:.4f}   | {vme:<10.2f} | {me:<10.2f}")

print("-" * 120)

results_df = pd.DataFrame(results)
if not results_df.empty:
    results_df.to_csv(OUTPUT_PATH, index=False)

    plot_df = results_df.sort_values(
        by=["Resistant_Count", "Test_Size"], ascending=[False, False]
    ).copy()

    fig, ax = plt.subplots(figsize=(7.5, 6))

    y_labels = plot_df.apply(
        lambda x: f"{x['Left_Out_Serovar']}  [N={int(x['Test_Size'])}, Res={int(x['Resistant_Count'])}]",
        axis=1,
    ).tolist()

    y_pos = np.arange(len(y_labels))
    bar_height = 0.35

    auc_scores = plot_df["ROC_AUC"].values
    f1_scores = plot_df["F1_Score"].values

    rects1 = ax.barh(
        y_pos - bar_height / 2,
        auc_scores,
        bar_height,
        label="ROC AUC",
        color="#56B4E9",
        edgecolor="none",
    )
    rects2 = ax.barh(
        y_pos + bar_height / 2,
        f1_scores,
        bar_height,
        label="F1 Score",
        color="#112E09",
        edgecolor="none",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()

    ax.set_xlabel("Performance Score (0.0 - 1.0)", labelpad=10)
    ax.set_title(
        "LOSO Validation Performance by Serovar Data Density", loc="center", pad=15
    )
    ax.set_xlim(0, 1.15)

    ax.axvline(
        0.5,
        color="gray",
        linestyle="--",
        alpha=0.7,
        zorder=0,
        label="Random (0.5)",
    )

    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)

    ax.bar_label(rects1, fmt="%.2f", padding=4)
    ax.bar_label(rects2, fmt="%.2f", padding=4)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.savefig(PLOT_PATH)
    plt.close()
    print(f"Saved correctly proportioned plot to {PLOT_PATH}")

else:
    print("\nNo valid serovars with mixed classes found to calculate AUC.")
