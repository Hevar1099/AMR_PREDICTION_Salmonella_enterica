import os
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score

os.makedirs("Output", exist_ok=True)

df = pd.read_csv("Data/final_ml_dataset.csv")
test_idx = np.load("Output/test_indices.npy", allow_pickle=True)
train_idx = df.index.difference(test_idx)

X = df.drop(columns=["label", "Assembly", "Serovar"])
y = df["label"]

X_train, y_train = X.loc[train_idx], y.loc[train_idx]
X_test, y_test = X.loc[test_idx], y.loc[test_idx]

print(f"Train set: N={len(y_train)} (R={int(y_train.sum())}, S={int((y_train == 0).sum())})")
print(f"Test set:  N={len(y_test)} (R={int(y_test.sum())}, S={int((y_test == 0).sum())})")

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


def calibration_intercept_slope(y_true, y_prob, eps=1e-6):
    p = np.clip(y_prob, eps, 1.0 - eps)
    logit_p = np.log(p / (1.0 - p))

    # Calibration-in-the-large (intercept with slope fixed at 1)
    citl_model = sm.GLM(
        y_true,
        np.ones((len(y_true), 1)),
        family=sm.families.Binomial(),
        offset=logit_p,
    ).fit()
    citl = float(citl_model.params[0])

    # Logistic recalibration intercept and slope
    X_cal = sm.add_constant(logit_p)
    cal_model = sm.GLM(y_true, X_cal, family=sm.families.Binomial()).fit()
    intercept = float(cal_model.params[0])
    slope = float(cal_model.params[1])
    return citl, intercept, slope


diag_rows = []
bin_rows = []
bin_edges = np.linspace(0.0, 1.0, 11)

for name, clf in models:
    y_prob = clf.predict_proba(X_test)[:, 1]
    citl, intercept, slope = calibration_intercept_slope(y_test.values, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    diag_rows.append(
        {
            "Model": name,
            "Test_N": len(y_test),
            "Test_R": int(y_test.sum()),
            "Test_S": int((y_test == 0).sum()),
            "ROC_AUC": round(float(auc), 4),
            "Brier": round(float(brier), 5),
            "CITL_Intercept": round(citl, 4),
            "Cal_Intercept": round(intercept, 4),
            "Cal_Slope": round(slope, 4),
            "Mean_Pred_Prob": round(float(np.mean(y_prob)), 4),
            "Obs_Prevalence": round(float(np.mean(y_test)), 4),
        }
    )

    for b in range(10):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b < 9:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        n_bin = int(mask.sum())
        pos_bin = int(y_test.values[mask].sum()) if n_bin > 0 else 0
        bin_rows.append(
            {
                "Model": name,
                "Bin": f"[{lo:.1f}, {hi:.1f}{']' if b == 9 else ')'}",
                "Count": n_bin,
                "Resistant": pos_bin,
                "Obs_Rate": round(pos_bin / n_bin, 4) if n_bin > 0 else np.nan,
                "Mean_Prob": round(float(y_prob[mask].mean()), 4) if n_bin > 0 else np.nan,
            }
        )

diag_df = pd.DataFrame(diag_rows)
bin_df = pd.DataFrame(bin_rows)

diag_df.to_csv("Output/calibration_diagnostics.csv", index=False)
bin_df.to_csv("Output/calibration_bin_counts.csv", index=False)

print("\nCalibration Diagnostics (Test Set):")
print(diag_df.to_string(index=False))
