import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBClassifier

os.makedirs("Output", exist_ok=True)

df = pd.read_csv("Data/final_ml_dataset.csv")
feat_cols = [c for c in df.columns if c not in ("label", "Assembly", "Serovar")]
X = df[feat_cols].astype(np.float64)
y = df["label"].values
groups = df["Serovar"].values


def bootstrap_auc_ci(y_t, y_p, n_boot=1000, seed=42):
    pos_idx = np.where(y_t == 1)[0]
    neg_idx = np.where(y_t == 0)[0]
    if len(pos_idx) < 2 or len(neg_idx) < 2:
        return np.nan, np.nan
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([b_pos, b_neg])
        vals.append(roc_auc_score(y_t[idx], y_p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


logo = LeaveOneGroupOut()
all_rows = []
low_auc_rows = []

for tr_idx, te_idx in logo.split(X, y, groups):
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    ser = groups[te_idx][0]

    n_tot = len(y_te)
    n_r = int(y_te.sum())
    n_s = n_tot - n_r

    # Nearest-neighbor Jaccard distance on 22-feature binary matrix (0 if both all-zero)
    dists = np.nan_to_num(cdist(X_te.values, X_tr.values, metric="jaccard"), nan=0.0)
    mean_jdist = float(np.mean(np.min(dists, axis=1)))

    # Train strictly inside fold (recompute scale_pos_weight on y_tr only)
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    clf = XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=spw,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    assert list(clf.classes_) == [0, 1], f"Unexpected class order in {ser}: {clf.classes_}"

    y_prob = clf.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_te, y_pred)
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1 = f1_score(y_te, y_pred) if (tp + fn) > 0 else np.nan
    vme = fn / (fn + tp) * 100 if (fn + tp) > 0 else np.nan
    me = fp / (fp + tn) * 100 if (fp + tn) > 0 else np.nan

    if n_r > 0 and n_s > 0:
        auc = float(roc_auc_score(y_te, y_prob))
        ci_lo, ci_hi = bootstrap_auc_ci(y_te, y_prob)
    else:
        auc, ci_lo, ci_hi = np.nan, np.nan, np.nan

    all_rows.append(
        {
            "Serovar": ser,
            "N": n_tot,
            "R": n_r,
            "S": n_s,
            "TP": int(tp),
            "FN": int(fn),
            "FP": int(fp),
            "TN": int(tn),
            "Accuracy": round(float(acc), 4),
            "Sensitivity": round(float(sens), 4) if not np.isnan(sens) else "NE",
            "Specificity": round(float(spec), 4) if not np.isnan(spec) else "NE",
            "Precision": round(float(prec), 4) if not np.isnan(prec) else "NE",
            "F1": round(float(f1), 4) if not np.isnan(f1) else "NE",
            "ROC_AUC": round(auc, 4) if not np.isnan(auc) else "NE",
            "AUC_CI_Lo": round(ci_lo, 4) if not np.isnan(ci_lo) else "NE",
            "AUC_CI_Hi": round(ci_hi, 4) if not np.isnan(ci_hi) else "NE",
            "VME_Pct": round(float(vme), 2) if not np.isnan(vme) else "NE",
            "ME_Pct": round(float(me), 2) if not np.isnan(me) else "NE",
            "Mean_Jaccard_Dist": round(mean_jdist, 5),
        }
    )

    if not np.isnan(auc) and auc < 0.5:
        gyra_vals = X_te["gyrA_D87Y"].values
        low_auc_rows.append(
            {
                "Serovar": ser,
                "N": n_tot,
                "R": n_r,
                "S": n_s,
                "ROC_AUC": round(auc, 4),
                "R_with_gyrA_D87Y": int(gyra_vals[y_te == 1].sum()),
                "S_with_gyrA_D87Y": int(gyra_vals[y_te == 0].sum()),
                "R_Prob_Mean": round(float(y_prob[y_te == 1].mean()), 4),
                "R_Prob_Min": round(float(y_prob[y_te == 1].min()), 4),
                "R_Prob_Max": round(float(y_prob[y_te == 1].max()), 4),
                "S_Prob_Mean": round(float(y_prob[y_te == 0].mean()), 4),
                "S_Prob_Min": round(float(y_prob[y_te == 0].min()), 4),
                "S_Prob_Max": round(float(y_prob[y_te == 0].max()), 4),
            }
        )

full_df = pd.DataFrame(all_rows).sort_values(["R", "N"], ascending=[False, False]).reset_index(drop=True)
full_df.to_csv("Output/loso_full_audit.csv", index=False)

two_class = full_df[(full_df["R"] > 0) & (full_df["S"] > 0)].copy()
two_class["ROC_AUC_num"] = two_class["ROC_AUC"].astype(float)

rho, p_spear = spearmanr(two_class["Mean_Jaccard_Dist"], two_class["ROC_AUC_num"])
r_pear, p_pear = pearsonr(two_class["Mean_Jaccard_Dist"], two_class["ROC_AUC_num"])

low_df = pd.DataFrame(low_auc_rows)
low_df.to_csv("Output/loso_low_auc_diagnostics.csv", index=False)

print("=== Two-Class Evaluable LOSO Serovars (13 / 93) ===")
cols_show = ["Serovar", "N", "R", "S", "TP", "FN", "FP", "TN", "ROC_AUC", "AUC_CI_Lo", "AUC_CI_Hi", "VME_Pct", "ME_Pct", "Mean_Jaccard_Dist"]
print(two_class[cols_show].to_string(index=False))

print("\n=== Distance vs. ROC AUC Correlation (13 two-class serovars) ===")
print(f"Spearman rho = {rho:.4f} (p = {p_spear:.4f})")
print(f"Pearson  r   = {r_pear:.4f} (p = {p_pear:.4f})")

print("\n=== Low-AUC (< 0.5) Score-Direction & gyrA_D87Y Diagnostics ===")
print(low_df.to_string(index=False))

# Single-class summary
all_s = full_df[full_df["R"] == 0]
all_r = full_df[full_df["S"] == 0]
print(f"\nSingle-class folds: {len(all_s)} all-susceptible (total N={all_s['N'].sum()}, total FP={all_s['FP'].sum()}, overall specificity={(all_s['TN'].sum()/all_s['N'].sum())*100:.2f}%); "
      f"{len(all_r)} all-resistant (Albert: N=1, TP={all_r['TP'].sum()}, FN={all_r['FN'].sum()})")
