import os
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)

os.makedirs("Output", exist_ok=True)

df = pd.read_csv("Data/final_ml_dataset.csv")
test_idx = np.load("Output/test_indices.npy", allow_pickle=True)
train_idx = df.index.difference(test_idx)

X = df.drop(columns=["label", "Assembly", "Serovar"])
y = df["label"]

X_train, y_train = X.loc[train_idx], y.loc[train_idx]
X_test, y_test = X.loc[test_idx], y.loc[test_idx]
y_true = y_test.values

spw = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Full dataset:  N={len(y)} (R={int(y.sum())} [{y.mean()*100:.2f}%], S={int((y==0).sum())} [{(1-y.mean())*100:.2f}%])")
print(f"Train subset:  N={len(y_train)} (R={int(y_train.sum())} [{y_train.mean()*100:.2f}%], S={int((y_train==0).sum())} [{(1-y_train.mean())*100:.2f}%])")
print(f"Test subset:   N={len(y_test)} (R={int(y_test.sum())} [{y_test.mean()*100:.2f}%], S={int((y_test==0).sum())} [{(1-y_test.mean())*100:.2f}%])")
print(f"scale_pos_weight (train): {spw:.4f}\n")

rf_uncal = joblib.load("Output/best_rf_model.pkl")
lr_uncal = joblib.load("Output/best_lr_model.pkl")
xgb_cal = joblib.load("Output/best_xgb_model.pkl")

xgb_uncal = xgb_cal.estimator
xgb_uncal.fit(X_train, y_train)

rf_cal = CalibratedClassifierCV(estimator=rf_uncal, method="isotonic", cv=5)
rf_cal.fit(X_train, y_train)

lr_cal = CalibratedClassifierCV(estimator=lr_uncal, method="isotonic", cv=5)
lr_cal.fit(X_train, y_train)

models = [
    ("RF_Uncalibrated", rf_uncal),
    ("RF_Calibrated", rf_cal),
    ("LR_Uncalibrated", lr_uncal),
    ("LR_Calibrated", lr_cal),
    ("XGB_Uncalibrated", xgb_uncal),
    ("XGB_Calibrated", xgb_cal),
]


def bootstrap_ci(y_t, y_p, metric_fn, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_t == 1)[0]
    neg_idx = np.where(y_t == 0)[0]
    vals = []
    for _ in range(n_boot):
        b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([b_pos, b_neg])
        vals.append(metric_fn(y_t[idx], y_p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def delong_roc_test(y_t, prob_a, prob_b):
    # Fast DeLong structural components for paired ROC AUC test
    pos_a, neg_a = prob_a[y_t == 1], prob_a[y_t == 0]
    pos_b, neg_b = prob_b[y_t == 1], prob_b[y_t == 0]
    m, n = len(pos_a), len(neg_a)

    def structural_components(pos, neg):
        # V10: for each positive, proportion of negatives with lower score (+0.5 ties)
        diff = pos[:, None] - neg[None, :]
        psi = (diff > 0).astype(float) + 0.5 * (diff == 0).astype(float)
        v10 = psi.mean(axis=1)
        v01 = psi.mean(axis=0)
        auc = v10.mean()
        return auc, v10, v01

    auc_a, v10_a, v01_a = structural_components(pos_a, neg_a)
    auc_b, v10_b, v01_b = structural_components(pos_b, neg_b)

    s10 = np.cov(np.vstack([v10_a, v10_b]))
    s01 = np.cov(np.vstack([v01_a, v01_b]))
    cov = s10 / m + s01 / n
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var_diff <= 0:
        return auc_a - auc_b, 0.0, 1.0
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = float(2 * stats.norm.sf(abs(z)))
    return float(auc_a - auc_b), float(z), p


probs_dict = {}
rows = []
y_prev = y_true.mean()
bs_naive = y_prev * (1.0 - y_prev)

for name, clf in models:
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    probs_dict[name] = y_prob

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    prec = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    roc_lo, roc_hi = bootstrap_ci(y_true, y_prob, roc_auc_score)
    pr_lo, pr_hi = bootstrap_ci(y_true, y_prob, average_precision_score)
    bs = brier_score_loss(y_true, y_prob)
    bss = 1.0 - bs / bs_naive
    vme = fn / (fn + tp) * 100
    me = fp / (fp + tn) * 100

    rows.append(
        {
            "Model": name,
            "Threshold": 0.50,
            "Test_N": len(y_true),
            "Test_R": int(y_true.sum()),
            "Test_S": int((y_true == 0).sum()),
            "TP": int(tp),
            "FN": int(fn),
            "FP": int(fp),
            "TN": int(tn),
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Sensitivity": round(float(sens), 4),
            "Specificity": round(float(spec), 4),
            "F1": round(float(f1), 4),
            "ROC_AUC": round(float(roc_auc), 4),
            "ROC_AUC_CI_Lo": round(roc_lo, 4),
            "ROC_AUC_CI_Hi": round(roc_hi, 4),
            "PR_AUC": round(float(pr_auc), 4),
            "PR_AUC_CI_Lo": round(pr_lo, 4),
            "PR_AUC_CI_Hi": round(pr_hi, 4),
            "Brier": round(float(bs), 5),
            "BSS": round(float(bss), 4),
            "VME_Pct": round(float(vme), 2),
            "ME_Pct": round(float(me), 3),
        }
    )

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv("Output/test_set_full_metrics.csv", index=False)

delong_rows = []
pairs = [
    ("XGB_Uncalibrated", "RF_Uncalibrated"),
    ("XGB_Uncalibrated", "LR_Uncalibrated"),
    ("RF_Uncalibrated", "LR_Uncalibrated"),
    ("XGB_Calibrated", "RF_Calibrated"),
    ("XGB_Calibrated", "LR_Calibrated"),
    ("RF_Calibrated", "LR_Calibrated"),
]
for m1, m2 in pairs:
    diff, z, p = delong_roc_test(y_true, probs_dict[m1], probs_dict[m2])
    delong_rows.append(
        {
            "Model_A": m1,
            "Model_B": m2,
            "AUC_Diff": round(diff, 4),
            "Z_Stat": round(z, 4),
            "P_Value": round(p, 4),
        }
    )

delong_df = pd.DataFrame(delong_rows)
delong_df.to_csv("Output/delong_comparisons.csv", index=False)

print(metrics_df[["Model", "TP", "FN", "FP", "TN", "Sensitivity", "Specificity", "Precision", "F1", "ROC_AUC", "PR_AUC", "VME_Pct", "ME_Pct"]].to_string(index=False))
print("\n95% Bootstrap CIs (1,000 iterations):")
print(metrics_df[["Model", "ROC_AUC", "ROC_AUC_CI_Lo", "ROC_AUC_CI_Hi", "PR_AUC", "PR_AUC_CI_Lo", "PR_AUC_CI_Hi"]].to_string(index=False))
print("\nDeLong Paired ROC AUC Tests:")
print(delong_df.to_string(index=False))
