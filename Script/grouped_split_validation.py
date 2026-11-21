import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

meta_sub = meta[["Assembly", "AMR genotypes"]].drop_duplicates(subset=["Assembly"])
merged = df.merge(meta_sub, on="Assembly", how="left")


def parse_genes(gene_str):
    if pd.isna(gene_str):
        return tuple()
    return tuple(
        sorted(set(g.split("=")[0].strip() for g in str(gene_str).split(",") if g.strip()))
    )


full_profiles = merged["AMR genotypes"].apply(parse_genes)
unique_profiles = full_profiles.unique()
prof_to_id = {p: i for i, p in enumerate(unique_profiles)}
prof_ids = np.array([prof_to_id[p] for p in full_profiles])

all_genes = sorted(set(g for p in unique_profiles for g in p))
gene_to_idx = {g: i for i, g in enumerate(all_genes)}
U = np.zeros((len(unique_profiles), len(all_genes)), dtype=np.uint8)
for i, p in enumerate(unique_profiles):
    for g in p:
        U[i, gene_to_idx[g]] = 1

X_full = pd.DataFrame(U[prof_ids], columns=all_genes, dtype=np.float64)
y = merged["label"].values
serovars = merged["Serovar"].values

dists = np.nan_to_num(pdist(U, metric="jaccard"), nan=0.0)
Z = linkage(dists, method="average")


def filter_features(X_train, y_train, serovar_train):
    variances = X_train.var()
    y_s = pd.Series(y_train, index=X_train.index)
    ser_s = pd.Series(serovar_train, index=X_train.index)
    keep = []
    for g in X_train.columns:
        if variances[g] < 0.01:
            continue
        ser_means = X_train[g].groupby(ser_s).mean()
        if (ser_means > 0.95).any():
            corr = abs(X_train[g].corr(y_s))
            if np.isnan(corr) or corr < 0.1:
                continue
        keep.append(g)
    return keep


schemes = {
    "Exact_Serovar_Genotype": pd.factorize(pd.Series(list(zip(serovars, prof_ids))))[0],
    "Jaccard_0.10": fcluster(Z, t=0.10, criterion="distance")[prof_ids],
    "Jaccard_0.20": fcluster(Z, t=0.20, criterion="distance")[prof_ids],
    "Serovar_Grouped": pd.factorize(serovars)[0],
}

rows = []
sgkf = StratifiedGroupKFold(n_splits=5)

for scheme_name, groups in schemes.items():
    n_groups = len(np.unique(groups))
    for model_name in ["LR", "RF", "XGB"]:
        fold_aucs, fold_aps, fold_f1s = [], [], []
        oof_prob = np.zeros(len(y))
        oof_pred = np.zeros(len(y), dtype=int)

        for tr_idx, te_idx in sgkf.split(X_full, y, groups):
            y_tr, y_te = y[tr_idx], y[te_idx]
            cols = filter_features(X_full.iloc[tr_idx], y_tr, serovars[tr_idx])
            X_tr = X_full.iloc[tr_idx][cols]
            X_te = X_full.iloc[te_idx][cols]

            spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
            if model_name == "LR":
                clf = LogisticRegression(
                    solver="liblinear",
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                )
            elif model_name == "RF":
                clf = RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                )
            else:
                clf = XGBClassifier(
                    eval_metric="logloss",
                    scale_pos_weight=spw,
                    random_state=42,
                    n_jobs=-1,
                )

            clf.fit(X_tr, y_tr)
            prob = clf.predict_proba(X_te)[:, 1]
            pred = clf.predict(X_te)
            oof_prob[te_idx] = prob
            oof_pred[te_idx] = pred

            if len(np.unique(y_te)) == 2:
                fold_aucs.append(roc_auc_score(y_te, prob))
                fold_aps.append(average_precision_score(y_te, prob))
                fold_f1s.append(f1_score(y_te, pred))

        tn, fp, fn, tp = confusion_matrix(y, oof_pred, labels=[0, 1]).ravel()
        vme = fn / (fn + tp) * 100
        me = fp / (fp + tn) * 100

        rows.append(
            {
                "Scheme": scheme_name,
                "Clusters": n_groups,
                "Model": model_name,
                "Fold_AUC_Mean": round(float(np.mean(fold_aucs)), 4),
                "Fold_AUC_SD": round(float(np.std(fold_aucs)), 4),
                "OOF_ROC_AUC": round(float(roc_auc_score(y, oof_prob)), 4),
                "OOF_PR_AUC": round(float(average_precision_score(y, oof_prob)), 4),
                "OOF_F1": round(float(f1_score(y, oof_pred)), 4),
                "TP": int(tp),
                "FN": int(fn),
                "FP": int(fp),
                "TN": int(tn),
                "VME_Pct": round(float(vme), 2),
                "ME_Pct": round(float(me), 2),
            }
        )

results_df = pd.DataFrame(rows)
results_df.to_csv("Output/grouped_split_results.csv", index=False)
print(results_df.to_string(index=False))
