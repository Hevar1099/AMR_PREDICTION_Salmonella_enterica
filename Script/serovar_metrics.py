import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV


df = pd.read_csv("Data/final_ml_dataset.csv")

X = df.drop(columns=["label", "Assembly", "Serovar"])
y = df["label"]
serovar = df["Serovar"]

# Split data
test_idx = np.load("Output/test_indices.npy", allow_pickle=True)
train_idx = df.index.difference(test_idx)

X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_train, y_test = y.loc[train_idx], y.loc[test_idx]
serovar_train, serovar_test = serovar.loc[train_idx], serovar.loc[test_idx]

scale_weight_pos = (y_train == 0).sum() / (y_train == 1).sum()

model = joblib.load("Output/best_xgb_model.pkl")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


test_df = pd.DataFrame(
    {"Serovar": serovar_test, "y_true": y_test, "y_pred": y_pred, "y_prob": y_prob}
)

serovars = ["Enteritidis", "Uganda", "Worthington", "Heidelberg", "Montevideo", "Hadar"]


results = []

for sero in serovars:
    df_sero = test_df[test_df["Serovar"] == sero]
    if len(df_sero) == 0:
        continue

    total_samples = len(df_sero)
    misclassifications = (df_sero["y_true"] != df_sero["y_pred"]).sum()
    error_rate = (misclassifications / total_samples) * 100

    tp = ((df_sero["y_true"] == 1) & (df_sero["y_pred"] == 1)).sum()
    fn = ((df_sero["y_true"] == 1) & (df_sero["y_pred"] == 0)).sum()
    fp = ((df_sero["y_true"] == 0) & (df_sero["y_pred"] == 1)).sum()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    brier_score = ((df_sero["y_prob"] - df_sero["y_true"]) ** 2).mean()

    results.append(
        {
            "Serovar": sero,
            "Total Samples": total_samples,
            "Lineage-Specific Error Rate (%)": error_rate,
            "Recall (Sensitivity)": recall,
            "Precision (Positive Predictive Value)": precision,
            "Calibration Error (Brier Score)": brier_score,
        }
    )


results_df = pd.DataFrame(results)
results_df.to_csv("Output/serovar_specific_metrics.csv", index=False)

print(
    "Serovar-specific metrics calculated and saved to Output/serovar_specific_metrics.csv"
)
print(results_df)
