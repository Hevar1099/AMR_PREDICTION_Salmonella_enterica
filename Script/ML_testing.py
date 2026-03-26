import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from sklearn.model_selection import (
    train_test_split,
    GroupKFold,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
)
import joblib
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
        "lines.markersize": 5,  # Let the global param control marker size!
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
        "figure.figsize": (3.5, 2.625),
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    }
)

os.makedirs("Output", exist_ok=True)
df = pd.read_csv("Data/final_ml_dataset.csv")

counts = df["label"].value_counts()
print(f"Label counts: {counts.to_dict()}")
print(f"Percentage: {counts / len(df) * 100}")

# 2. Split Data
X = df.drop(columns=["label", "Assembly", "Serovar"])
y = df["label"]

# Ensure each serovar appears in the test set at least once.
# Strategy: reserve one random sample per serovar, then stratify the remaining split by label.
rng = np.random.RandomState(42)
serovar_to_index = df.groupby("Serovar").apply(lambda g: rng.choice(g.index, size=1)[0])
serovar_holdout_idx = serovar_to_index.values

desired_test_size = int(len(df) * 0.2)
remaining_test = max(desired_test_size - len(serovar_holdout_idx), 0)

remaining_idx = df.index.difference(serovar_holdout_idx)
X_remaining = X.loc[remaining_idx]
y_remaining = y.loc[remaining_idx]

if remaining_test > 0:
    _, X_test_extra, _, y_test_extra = train_test_split(
        X_remaining,
        y_remaining,
        test_size=remaining_test / len(X_remaining),
        random_state=42,
        stratify=y_remaining,
    )
    test_idx = np.concatenate([serovar_holdout_idx, X_test_extra.index.values])
else:
    test_idx = serovar_holdout_idx

train_idx = df.index.difference(test_idx)

X_train = X.loc[train_idx]
X_test = X.loc[test_idx]
y_train = y.loc[train_idx]
y_test = y.loc[test_idx]

# Verify test set independence
train_assemblies = set(df.loc[train_idx, "Assembly"])
test_assemblies = set(df.loc[test_idx, "Assembly"])
overlap = train_assemblies & test_assemblies
if overlap:
    print(f"WARNING: Overlapping assemblies: {overlap}")
else:
    print("Test set independence verified: No overlapping assemblies.")

# Check serovar distribution
print("Train serovar distribution:")
print(df.loc[train_idx, "Serovar"].value_counts())
print("Test serovar distribution:")
print(df.loc[test_idx, "Serovar"].value_counts())

scale_weight_pos = (y_train == 0).sum() / (y_train == 1).sum()

# Hyperparameter Tuning
xgb_base = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=scale_weight_pos,
    random_state=42,
)

xgb_param_dist = {
    "estimator__max_depth": [3, 5, 7, 9, 12],
    "estimator__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "estimator__subsample": [0.5, 0.7, 0.9, 1.0],
    "estimator__gamma": [0, 0.5, 1.0, 2.0, 5.0],
}

print("\nRunning RandomizedSearchCV for Calibrated XGBoost hyperparameters...")
cal_xgb_factory = CalibratedClassifierCV(estimator=xgb_base, method="isotonic", cv=5)

xgb_search = RandomizedSearchCV(
    estimator=cal_xgb_factory,
    param_distributions=xgb_param_dist,
    n_iter=20,
    scoring="roc_auc",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
xgb_search.fit(X_train, y_train)
print("Best XGBoost params:", xgb_search.best_params_)
print("Best XGBoost CV ROC AUC:", xgb_search.best_score_)

best_calibrated_xgb = xgb_search.best_estimator_

# Definitions for the evaluation loop
models_to_train = {
    "RF": RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    ),
    "LR": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "XGB": best_calibrated_xgb,
}


def get_statistical_rigor(y_true, y_prob):
    # Brier Decomposition & BSS
    y_avg = np.mean(y_true)
    uncertainty = y_avg * (1 - y_avg)
    bs = np.mean((y_prob - y_true) ** 2)
    bss = 1 - (bs / uncertainty) if uncertainty > 0 else 0

    # Reliability & Resolution (Simplified for performance)
    p_true, p_pred = calibration_curve(y_true, y_prob, n_bins=10)
    reliability = np.mean((p_true - p_pred) ** 2)
    return bs, bss, reliability, uncertainty


def calculate_net_benefit(y_true, y_prob, thresholds):
    net_benefit = []
    for t in thresholds:
        tp = np.sum((y_prob >= t) & (y_true == 1))
        fp = np.sum((y_prob >= t) & (y_true == 0))
        nb = (tp / len(y_true)) - (fp / len(y_true)) * (t / (1 - t)) if t < 1 else 0
        net_benefit.append(nb)
    return net_benefit


# 4. Train and Evaluate
all_calibrated_probs = {}
results = []

# --- Academic Styling ---
colors = ["#E69F00", "#56B4E9", "#009E73"]
fig = plt.figure(figsize=(7.5, 6))
gs = fig.add_gridspec(2, 2, height_ratios=[4, 1.2], hspace=0.3, wspace=0.25)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax1_dist = fig.add_subplot(gs[1, 0], sharex=ax1)
ax2_dist = fig.add_subplot(gs[1, 1], sharex=ax2)

# Reference lines
for ax in [ax1, ax2]:
    ax.plot(
        [0, 1],
        [0, 1],
        color="grey",
        linestyle=(0, (5, 5)),
        linewidth=1.5,
        label="Perfectly Calibrated",
        zorder=1,
    )

for i, (name, model) in enumerate(models_to_train.items()):
    print(f"\nProcessing {name}...")
    color = colors[i]

    # --- A. Uncalibrated ---
    if name == "XGB":
        base_model = model.estimator
    else:
        base_model = model

    base_model.fit(X_train, y_train)
    y_prob_uncal = base_model.predict_proba(X_test)[:, 1]
    y_pred_uncal = base_model.predict(X_test)

    # Calculate Clinical Errors
    tn_un, fp_un, fn_un, tp_un = confusion_matrix(
        y_test, y_pred_uncal, labels=[0, 1]
    ).ravel()
    vme_un = (fn_un / (fn_un + tp_un)) * 100 if (fn_un + tp_un) > 0 else 0
    me_un = (fp_un / (fp_un + tn_un)) * 100 if (fp_un + tn_un) > 0 else 0
    bs, bss, rel, unc = get_statistical_rigor(y_test, y_prob_uncal)
    results.append(
        {
            "Model": f"{name}_Uncalibrated",
            "Accuracy": accuracy_score(y_test, y_pred_uncal),
            "Precision": precision_score(y_test, y_pred_uncal),
            "Recall": recall_score(y_test, y_pred_uncal),
            "F1-Score": f1_score(y_test, y_pred_uncal),
            "ROC AUC": roc_auc_score(y_test, y_prob_uncal),
            "Brier": bs,
            "Brier Skill Score": bss,
            "Reliability": rel,
            "VME": vme_un,
            "ME": me_un,
        }
    )

    p_true_un, p_pred_un = calibration_curve(y_test, y_prob_uncal, n_bins=10)
    ax1.plot(
        p_pred_un,
        p_true_un,
        marker="s",
        linestyle="--",
        color=color,
        label=f"{name}",
        alpha=0.8,
        zorder=2,
    )

    ax1_dist.hist(
        y_prob_uncal,
        range=(0, 1),
        bins=20,
        color=color,
        histtype="step",
        alpha=0.7,
    )

    # --- B. Calibrated ---
    if name == "XGB":
        cal_model = model
    else:
        cal_model = CalibratedClassifierCV(
            estimator=base_model, method="isotonic", cv=5
        )
        cal_model.fit(X_train, y_train)

    y_prob = cal_model.predict_proba(X_test)[:, 1]
    all_calibrated_probs[name] = y_prob
    y_pred_cal = cal_model.predict(X_test)

    tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_test, y_pred_cal, labels=[0, 1]).ravel()
    vme_c = (fn_c / (fn_c + tp_c)) * 100 if (fn_c + tp_c) > 0 else 0
    me_c = (fp_c / (fp_c + tn_c)) * 100 if (fp_c + tn_c) > 0 else 0

    bs_c, bss_c, rel_c, unc_c = get_statistical_rigor(y_test, y_prob)
    results.append(
        {
            "Model": f"{name}_Calibrated",
            "Accuracy": accuracy_score(y_test, y_pred_cal),
            "Precision": precision_score(y_test, y_pred_cal),
            "Recall": recall_score(y_test, y_pred_cal),
            "F1-Score": f1_score(y_test, y_pred_cal),
            "ROC AUC": roc_auc_score(y_test, y_prob),
            "Brier": bs_c,
            "Brier Skill Score": bss_c,
            "Reliability": rel_c,
            "VME": vme_c,
            "ME": me_c,
        }
    )

    p_true, p_pred = calibration_curve(y_test, y_prob, n_bins=10)
    line = ax2.plot(
        p_pred,
        p_true,
        marker="o",
        color=color,
        label=f"{name}",
        zorder=3,
    )

    ax2_dist.hist(
        y_prob, range=(0, 1), bins=20, color=color, histtype="step", alpha=0.7
    )

# --- Final Refinements ---
ax1.set_title("A: Base Models (Uncalibrated)", loc="left")
ax2.set_title("B: Isotonic Calibration", loc="left")

for ax in [ax1, ax2]:
    ax.set_ylabel("Observed Fraction of Positives")
    ax.legend(frameon=False, loc="upper left")

for ax in [ax1_dist, ax2_dist]:
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)

plt.tight_layout()
plt.savefig("Output/Figure8_Calibration.png", dpi=600, bbox_inches="tight")
plt.show()

# --- Decision Curve Analysis (DCA) ---
thresh = np.linspace(0.01, 0.99, 100)
plt.figure(figsize=(8, 3))

# Baseline: Treat All
y_prev = np.mean(y_test)
nb_all = (y_prev) - (1 - y_prev) * (thresh / (1 - thresh))
plt.plot(thresh, nb_all, color="black", linestyle="--", alpha=0.3, label="Treat All")
plt.axhline(0, color="grey", lw=1, label="Treat None")

# Plot each model from our collected dictionary
for i, (name, probs) in enumerate(all_calibrated_probs.items()):
    nb_model = calculate_net_benefit(y_test, probs, thresh)
    plt.plot(thresh, nb_model, color=okabe_ito_palette[i], label=f"{name} (Calibrated)")

plt.xlim(0.0, 1.0)

plt.ylim(-0.01, y_prev * 1.2)

plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.legend(frameon=False)

plt.tight_layout()
plt.savefig("Output/Figure11_DCA.png", bbox_inches="tight")
plt.show()

# Feature Importance Extraction
print("\nExtracting Feature Importance...")

all_importances = []
for clf in best_calibrated_xgb.calibrated_classifiers_:
    if hasattr(clf, "estimator"):
        all_importances.append(clf.estimator.feature_importances_)
    else:
        all_importances.append(clf.base_estimator.feature_importances_)

importances_mean = np.mean(all_importances, axis=0)
importances_std = np.std(all_importances, axis=0)

feat_imp_df = (
    pd.DataFrame(
        {
            "Feature": X_train.columns,
            "Importance": importances_mean,
            "Std": importances_std,
        }
    )
    .sort_values(by="Importance", ascending=False)
    .head(20)
)

plt.figure(figsize=(5, 6))
plt.barh(
    y=feat_imp_df["Feature"],
    width=feat_imp_df["Importance"],
    xerr=feat_imp_df["Std"],
    color="#0072B2",  # Okabe-Ito Blue
    edgecolor="black",
    capsize=3,
)

plt.gca().invert_yaxis()
plt.xlabel("Mean Importance")
plt.tight_layout()

plt.savefig("Output/feature_importance.png", dpi=600, bbox_inches="tight")
plt.show()
plt.close()

results_df = pd.DataFrame(results).set_index("Model")
results_df.to_csv("Output/model_calibration_comparison.csv")
print("\nModel Comparison (Uncalibrated vs Calibrated):")
print(results_df)

serovar_counts = df["Serovar"].value_counts()
valid_serovars = serovar_counts[serovar_counts >= 10].index
df_blocked = df[df["Serovar"].isin(valid_serovars)].copy()

y_blocked = df_blocked["label"]
groups = df_blocked["Serovar"]

cols_to_drop = [
    "label",
    "Assembly",
    "Serovar",
    "#BioSample",
    "AMR genotypes",
    "AST phenotypes",
    "cipro_status",
]
existing_drop_cols = [c for c in cols_to_drop if c in df_blocked.columns]
X_blocked = df_blocked.drop(columns=existing_drop_cols).select_dtypes(
    include=[np.number]
)

print(
    f"\nRunning Blocked CV for all models (Raw vs Calibrated) on {X_blocked.shape[1]} features..."
)
gkf = GroupKFold(n_splits=5)

cv_stability_results = []

for name, model in models_to_train.items():
    if name == "XGB":
        uncal_m = model.estimator
        cal_m = model
    else:
        uncal_m = model
        cal_m = CalibratedClassifierCV(estimator=model, method="isotonic", cv=5)

    # Run CV for Uncalibrated
    print(f"  CV for {name} Uncalibrated...")
    uncal_scores = cross_val_score(
        uncal_m, X_blocked, y_blocked, groups=groups, cv=gkf, scoring="roc_auc"
    )
    for fold_idx, score in enumerate(uncal_scores):
        cv_stability_results.append(
            {"Model": f"{name}_Uncalibrated", "ROC AUC": score, "Fold": fold_idx}
        )

    # Run CV for Calibrated
    print(f"  CV for {name} Calibrated...")
    cal_scores = cross_val_score(
        cal_m, X_blocked, y_blocked, groups=groups, cv=gkf, scoring="roc_auc"
    )
    for fold_idx, score in enumerate(cal_scores):
        cv_stability_results.append(
            {"Model": f"{name}_Calibrated", "ROC AUC": score, "Fold": fold_idx}
        )

cv_stability_df = pd.DataFrame(cv_stability_results)
cv_stability_df.to_csv("Output/model_comparison.csv", index=False)

print("\n" + "=" * 40)
print("DETAILED CV RESULTS SAVED TO: Output/model_comparison.csv")
print("=" * 40)
print(cv_stability_df.groupby("Model")["ROC AUC"].mean())

rf_final = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight="balanced"
)
rf_final.fit(X_train, y_train)
joblib.dump(rf_final, "Output/best_rf_model.pkl")

lr_final = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr_final.fit(X_train, y_train)
joblib.dump(lr_final, "Output/best_lr_model.pkl")

joblib.dump(best_calibrated_xgb, "Output/best_xgb_model.pkl")
np.save("Output/test_indices.npy", test_idx)
