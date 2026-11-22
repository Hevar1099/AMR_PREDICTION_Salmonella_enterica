import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

os.makedirs("Output", exist_ok=True)

df = pd.read_csv("Data/final_ml_dataset.csv")
test_idx = np.load("Output/test_indices.npy", allow_pickle=True)
train_idx = df.index.difference(test_idx)

X = df.drop(columns=["label", "Assembly", "Serovar"])
y = df["label"]
feats = X.columns.tolist()

X_train, X_test = X.loc[train_idx], X.loc[test_idx]
model = joblib.load("Output/best_xgb_model.pkl")

gain_folds = []
shap_train_folds = []
shap_test_folds = []

dtrain = xgb.DMatrix(X_train, feature_names=feats)
dtest = xgb.DMatrix(X_test, feature_names=feats)

for cal_clf in model.calibrated_classifiers_:
    est = cal_clf.estimator
    gain_folds.append(est.feature_importances_)
    booster = est.get_booster()
    sv_tr = booster.predict(dtrain, pred_contribs=True)[:, :-1]
    sv_te = booster.predict(dtest, pred_contribs=True)[:, :-1]
    shap_train_folds.append(np.abs(sv_tr).mean(axis=0))
    shap_test_folds.append(np.abs(sv_te).mean(axis=0))

gain_folds = np.array(gain_folds)
shap_train_folds = np.array(shap_train_folds)
shap_test_folds = np.array(shap_test_folds)

summary_df = pd.DataFrame(
    {
        "Feature": feats,
        "XGB_Gain_Mean": np.round(gain_folds.mean(axis=0), 4),
        "XGB_Gain_SD": np.round(gain_folds.std(axis=0), 4),
        "Mean_Abs_SHAP_Train": np.round(shap_train_folds.mean(axis=0), 4),
        "SD_Abs_SHAP_Train": np.round(shap_train_folds.std(axis=0), 4),
        "Mean_Abs_SHAP_Test": np.round(shap_test_folds.mean(axis=0), 4),
        "SD_Abs_SHAP_Test": np.round(shap_test_folds.std(axis=0), 4),
    }
).sort_values("Mean_Abs_SHAP_Test", ascending=False)

summary_df.to_csv("Output/shap_stability_summary.csv", index=False)

# Pairwise Spearman rank correlations across the 5 CV folds (on X_test)
fold_corrs = []
for i in range(len(shap_test_folds)):
    for j in range(i + 1, len(shap_test_folds)):
        r, _ = spearmanr(shap_test_folds[i], shap_test_folds[j])
        fold_corrs.append(r)

tr_te_r, _ = spearmanr(shap_train_folds.mean(axis=0), shap_test_folds.mean(axis=0))

print("Top 10 features by Mean |SHAP| (Test vs Train) and XGBoost Gain:")
print(summary_df.head(10).to_string(index=False))
print(f"\nSpearman rank correlation across 5 CV folds (test): mean={np.mean(fold_corrs):.4f}, min={np.min(fold_corrs):.4f}")
print(f"Spearman rank correlation (train vs test mean |SHAP|): {tr_te_r:.4f}")

# Check all quinolone-related determinants in raw AMR genotypes (for the 8,759 isolates)
meta = pd.read_csv("Data/cleaned_metadata.csv")
meta_sub = meta[["Assembly", "AMR genotypes"]].drop_duplicates(subset=["Assembly"])
merged = df[["Assembly", "label"]].merge(meta_sub, on="Assembly", how="left")

genes = merged["AMR genotypes"].apply(
    lambda s: [] if pd.isna(s) else [g.split("=")[0].strip() for g in str(s).split(",") if g.strip()]
)
gm = genes.str.join("|").str.get_dummies()

q_keywords = ("gyr", "par", "qnr", "oqx", "aac(6')-ib-cr", "qepa")
q_cols = [c for c in gm.columns if any(k in c.lower() for k in q_keywords)]
q_rows = []
for c in sorted(q_cols):
    cnt = int(gm[c].sum())
    var = float(gm[c].var())
    r_cnt = int(((gm[c] == 1) & (merged["label"] == 1)).sum())
    q_rows.append(
        {
            "Determinant": c,
            "Total_Present": cnt,
            "Prevalence_Pct": round(cnt / len(gm) * 100, 3),
            "Variance": round(var, 5),
            "Resistant_With_Det": r_cnt,
            "Retained_In_22": c in feats,
        }
    )

q_df = pd.DataFrame(q_rows).sort_values("Total_Present", ascending=False)
q_df.to_csv("Output/raw_quinolone_determinants.csv", index=False)
print("\nQuinolone-related determinants in raw AMRFinderPlus annotations (N=8,759):")
print(q_df.to_string(index=False))
