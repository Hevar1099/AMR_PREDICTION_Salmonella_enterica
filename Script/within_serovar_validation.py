import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
)

os.makedirs("Output", exist_ok=True)

meta = pd.read_csv("Data/cleaned_metadata.csv")
df = pd.read_csv("Data/final_ml_dataset.csv")
test_idx = np.load("Output/test_indices.npy", allow_pickle=True)

meta_sub = meta[["Assembly", "AMR genotypes"]].drop_duplicates(subset=["Assembly"])
merged = df.merge(meta_sub, on="Assembly", how="left")

feat_cols = [c for c in df.columns if c not in ("Assembly", "label", "Serovar")]
target_serovars = ["Infantis", "Enteritidis", "Monophasic Variant"]

# 1. Within-serovar evaluation on the untouched 80/20 test set using saved models
rf_model = joblib.load("Output/best_rf_model.pkl")
lr_model = joblib.load("Output/best_lr_model.pkl")
xgb_model = joblib.load("Output/best_xgb_model.pkl")
saved_models = {"RF": rf_model, "LR": lr_model, "XGB": xgb_model}

rows = []
df_test = df.loc[test_idx]

for ser in target_serovars:
    sub = df_test[df_test["Serovar"] == ser]
    X_s = sub[feat_cols]
    y_s = sub["label"].values
    n_tot, n_r, n_s = len(y_s), int(y_s.sum()), int((y_s == 0).sum())
    for mname, clf in saved_models.items():
        prob = clf.predict_proba(X_s)[:, 1]
        pred = clf.predict(X_s)
        tn, fp, fn, tp = confusion_matrix(y_s, pred, labels=[0, 1]).ravel()
        auc = roc_auc_score(y_s, prob) if len(np.unique(y_s)) == 2 else np.nan
        ap = average_precision_score(y_s, prob) if len(np.unique(y_s)) == 2 else np.nan
        f1 = f1_score(y_s, pred) if len(np.unique(y_s)) == 2 else np.nan
        rows.append(
            {
                "Analysis": "Main_Test_Set_Within_Serovar",
                "Serovar": ser,
                "Model": mname,
                "N": n_tot,
                "R": n_r,
                "S": n_s,
                "ROC_AUC": round(float(auc), 4),
                "PR_AUC": round(float(ap), 4),
                "F1": round(float(f1), 4),
                "TP": int(tp),
                "FN": int(fn),
                "FP": int(fp),
                "TN": int(tn),
            }
        )

# 2. Dedicated within-serovar 5-fold StratifiedKFold (trained & tested strictly inside each serovar)
for ser in target_serovars:
    sub = merged[merged["Serovar"] == ser].reset_index(drop=True)
    X_s = sub[feat_cols]
    y_s = sub["label"].values
    n_tot, n_r, n_s = len(y_s), int(y_s.sum()), int((y_s == 0).sum())

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for mname in ["LR", "RF", "XGB"]:
        oof_prob = np.zeros(n_tot)
        oof_pred = np.zeros(n_tot, dtype=int)
        for tr_i, te_i in skf.split(X_s, y_s):
            X_tr, X_te = X_s.iloc[tr_i], X_s.iloc[te_i]
            y_tr = y_s[tr_i]
            spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
            if mname == "LR":
                clf = LogisticRegression(
                    solver="liblinear", max_iter=1000, class_weight="balanced", random_state=42
                )
            elif mname == "RF":
                clf = RandomForestClassifier(
                    n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
                )
            else:
                clf = XGBClassifier(
                    eval_metric="logloss", scale_pos_weight=spw, random_state=42, n_jobs=-1
                )
            clf.fit(X_tr, y_tr)
            oof_prob[te_i] = clf.predict_proba(X_te)[:, 1]
            oof_pred[te_i] = clf.predict(X_te)

        tn, fp, fn, tp = confusion_matrix(y_s, oof_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "Analysis": "Within_Serovar_5Fold_CV",
                "Serovar": ser,
                "Model": mname,
                "N": n_tot,
                "R": n_r,
                "S": n_s,
                "ROC_AUC": round(float(roc_auc_score(y_s, oof_prob)), 4),
                "PR_AUC": round(float(average_precision_score(y_s, oof_prob)), 4),
                "F1": round(float(f1_score(y_s, oof_pred)), 4),
                "TP": int(tp),
                "FN": int(fn),
                "FP": int(fp),
                "TN": int(tn),
            }
        )

# 3. Within-S. Infantis genotype-grouped 5-fold CV (no identical full AMR genotype split across train/test)
inf = merged[merged["Serovar"] == "Infantis"].reset_index(drop=True)
X_inf = inf[feat_cols]
y_inf = inf["label"].values
prof_groups = pd.factorize(inf["AMR genotypes"].fillna("NONE"))[0]
sgkf = StratifiedGroupKFold(n_splits=5)

for mname in ["LR", "RF", "XGB"]:
    oof_prob = np.zeros(len(y_inf))
    oof_pred = np.zeros(len(y_inf), dtype=int)
    for tr_i, te_i in sgkf.split(X_inf, y_inf, prof_groups):
        X_tr, X_te = X_inf.iloc[tr_i], X_inf.iloc[te_i]
        y_tr = y_inf[tr_i]
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        if mname == "LR":
            clf = LogisticRegression(
                solver="liblinear", max_iter=1000, class_weight="balanced", random_state=42
            )
        elif mname == "RF":
            clf = RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
            )
        else:
            clf = XGBClassifier(
                eval_metric="logloss", scale_pos_weight=spw, random_state=42, n_jobs=-1
            )
        clf.fit(X_tr, y_tr)
        oof_prob[te_i] = clf.predict_proba(X_te)[:, 1]
        oof_pred[te_i] = clf.predict(X_te)

    tn, fp, fn, tp = confusion_matrix(y_inf, oof_pred, labels=[0, 1]).ravel()
    rows.append(
        {
            "Analysis": "Within_Infantis_Genotype_Grouped_5Fold",
            "Serovar": "Infantis",
            "Model": mname,
            "N": len(y_inf),
            "R": int(y_inf.sum()),
            "S": int((y_inf == 0).sum()),
            "ROC_AUC": round(float(roc_auc_score(y_inf, oof_prob)), 4),
            "PR_AUC": round(float(average_precision_score(y_inf, oof_prob)), 4),
            "F1": round(float(f1_score(y_inf, oof_pred)), 4),
            "TP": int(tp),
            "FN": int(fn),
            "FP": int(fp),
            "TN": int(tn),
        }
    )

res_df = pd.DataFrame(rows)
res_df.to_csv("Output/within_serovar_results.csv", index=False)
print(res_df.to_string(index=False))
