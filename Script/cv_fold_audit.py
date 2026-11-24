import os
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

os.makedirs("Output", exist_ok=True)

df = pd.read_csv("Data/final_ml_dataset.csv")
serovar_counts = df["Serovar"].value_counts()
valid_serovars = serovar_counts[serovar_counts >= 10].index
df_blocked = df[df["Serovar"].isin(valid_serovars)].copy().reset_index(drop=True)

y_blocked = df_blocked["label"]
groups = df_blocked["Serovar"]
X_blocked = df_blocked.drop(columns=["label", "Assembly", "Serovar"])

gkf = GroupKFold(n_splits=5)
fold_rows = []

for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X_blocked, y_blocked, groups)):
    sub = df_blocked.iloc[te_idx]
    n_tot = len(sub)
    n_r = int(sub["label"].sum())
    n_s = n_tot - n_r
    ser_r = sub.groupby("Serovar")["label"].agg(["count", "sum"])
    res_sers = ser_r[ser_r["sum"] > 0].sort_values("sum", ascending=False)
    res_str = "; ".join(f"{s}({int(r['sum'])}/{int(r['count'])})" for s, r in res_sers.iterrows())
    fold_rows.append(
        {
            "Fold": fold_idx,
            "Test_N": n_tot,
            "Test_R": n_r,
            "Test_S": n_s,
            "R_Pct": round(n_r / n_tot * 100, 2),
            "N_Serovars": int(sub["Serovar"].nunique()),
            "Resistant_Serovars": res_str,
        }
    )

    if fold_idx == 1:
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        lr.fit(X_blocked.iloc[tr_idx], y_blocked.iloc[tr_idx])
        p_uncal = lr.predict_proba(X_blocked.iloc[te_idx])[:, 1]
        lr_cal = CalibratedClassifierCV(estimator=lr, method="isotonic", cv=5)
        lr_cal.fit(X_blocked.iloc[tr_idx], y_blocked.iloc[tr_idx])
        p_cal = lr_cal.predict_proba(X_blocked.iloc[te_idx])[:, 1]
        print("Fold 1 LR diagnostic:")
        print(f"  Uncalibrated AUC: {roc_auc_score(y_blocked.iloc[te_idx], p_uncal):.4f}")
        print(f"  Calibrated AUC:   {roc_auc_score(y_blocked.iloc[te_idx], p_cal):.4f}")
        print("  Fold 1 resistant isolates (Serovar, uncal_prob, cal_prob):")
        pos_mask = y_blocked.iloc[te_idx].values == 1
        for s_name, pu, pc in zip(sub["Serovar"].values[pos_mask], p_uncal[pos_mask], p_cal[pos_mask]):
            print(f"    {s_name:<18} uncal={pu:.4f}  cal={pc:.4f}")
        print(f"  Fold 1 susceptible unique cal_probs (top 5): {pd.Series(p_cal[~pos_mask]).round(4).value_counts().head(5).to_dict()}\n")

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv("Output/cv_fold_composition.csv", index=False)

cv_df = pd.read_csv("Output/model_comparison.csv")
summary_df = (
    cv_df.groupby("Model")["ROC AUC"]
    .agg(Mean="mean", SD="std", Min="min", Max="max")
    .round(4)
    .reset_index()
)
summary_df.to_csv("Output/cv_model_summary.csv", index=False)

print("5-Fold GroupKFold Composition (serovars with N >= 10):")
print(fold_df.to_string(index=False))
print("\nModel ROC AUC Summary across 5 Folds:")
print(summary_df.to_string(index=False))
