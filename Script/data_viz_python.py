import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import shap
import matplotlib
import os
import xgboost as xgb
from xgboost import XGBClassifier
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from cycler import cycler

# --- Global Colors and Fonts ---
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
OUTPUT_DIR = "Output/"
MODEL_COMPARISON_PATH = f"{OUTPUT_DIR}model_comparison.csv"


# ==========================================
# Helper: Train Model on the Fly
# ==========================================
def get_live_model_and_data(df):
    print("\n[System] Loading saved models and test indices...")

    cols_to_drop = [
        "label",
        "Assembly",
        "Serovar",
        "#BioSample",
        "AMR genotypes",
        "AST phenotypes",
        "cipro_status",
    ]
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).select_dtypes(
        include=[np.number]
    )
    y = df["label"]

    test_idx = np.load("Output/test_indices.npy", allow_pickle=True)
    train_idx = df.index.difference(test_idx)

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    xgb_model = joblib.load("Output/best_xgb_model.pkl")
    rf_model = joblib.load("Output/best_rf_model.pkl")
    lr_model = joblib.load("Output/best_lr_model.pkl")

    comparison_models = {
        "XGBoost": xgb_model,
        "Random Forest": rf_model,
        "Logistic Regression": lr_model,
    }

    return (
        xgb_model,
        X_train,
        X_test,
        y_train,
        y_test,
        X.columns.tolist(),
        comparison_models,
    )


# ==========================================
# 1. Table 1: Demographics
# ==========================================
def generate_table1(df):
    print("Generating Table 1...")
    summary = df.groupby("Serovar")["label"].agg(["count", "sum"]).reset_index()
    summary.columns = ["Serovar", "Total_N", "Resistant_N"]
    summary["Susceptible_N"] = summary["Total_N"] - summary["Resistant_N"]
    summary["Resistant_%"] = (summary["Resistant_N"] / summary["Total_N"] * 100).round(
        1
    )
    summary["Susceptible_%"] = (
        summary["Susceptible_N"] / summary["Total_N"] * 100
    ).round(1)

    summary["Ciprofloxacin Resistant"] = summary.apply(
        lambda x: f"{int(x['Resistant_N'])} ({x['Resistant_%']}%)", axis=1
    )
    summary["Ciprofloxacin Susceptible"] = summary.apply(
        lambda x: f"{int(x['Susceptible_N'])} ({x['Susceptible_%']}%)", axis=1
    )

    final_table = summary.sort_values("Total_N", ascending=False)[
        ["Serovar", "Total_N", "Ciprofloxacin Resistant", "Ciprofloxacin Susceptible"]
    ].head(15)
    final_table.to_csv(f"{OUTPUT_DIR}Table1_demographics.csv", index=False)
    print(f"Saved: {OUTPUT_DIR}Table1_demographics.csv")


# ==========================================
# 2. Figure 1: Treemap
# ==========================================
def plot_serovar_treemap(df):
    print("Generating Figure 1 (Treemap)...")
    serovar_stats = df.groupby("Serovar")["label"].agg(["count", "mean"]).reset_index()
    serovar_stats.columns = ["Serovar", "N", "Resistance_Rate"]

    top_serovars = serovar_stats.sort_values("N", ascending=False).head(25)

    cmap = matplotlib.colormaps.get_cmap("RdYlBu_r")
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    colors = [cmap(norm(val)) for val in top_serovars["Resistance_Rate"]]

    plt.figure(figsize=(7.5, 5.5))
    ax = plt.gca()

    squarify.plot(
        sizes=top_serovars["N"],
        label=top_serovars.apply(lambda x: f"{x['Serovar']}\n(n={x['N']})", axis=1),
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=1.0,
        text_kwargs={"fontsize": 6, "color": "black"},
        ax=ax,
    )

    plt.title("Landscape of Salmonella Serovars")
    plt.axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation="vertical", pad=0.02, ax=ax)
    cbar.set_label("Proportion Resistant (0.0 - 1.0)", rotation=270, labelpad=20)

    plt.savefig(f"{OUTPUT_DIR}Figure1_Serovar_Treemap.png")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}Figure1_Serovar_Treemap.png")


# ==========================================
# 3. Figure 2: ROC/PR Curves
# ==========================================
def plot_performance_curves(comparison_models, X_test, y_test):
    print("Generating Figure 2 (ROC/PR Curves)...")

    xgb_model = comparison_models["XGBoost"]
    rf_model = comparison_models["Random Forest"]
    lr_model = comparison_models["Logistic Regression"]

    y_probs_xgb = xgb_model.predict_proba(X_test)[:, 1]
    y_probs_lr = lr_model.predict_proba(X_test)[:, 1]
    y_probs_rf = rf_model.predict_proba(X_test)[:, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5))
    plot_colors = okabe_ito_palette[:3]

    # ROC
    for (name, probs), color in zip(
        [
            ("XGBoost", y_probs_xgb),
            ("Logistic Regression", y_probs_lr),
            ("Random Forest", y_probs_rf),
        ],
        plot_colors,
    ):
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=color, label=f"{name} (AUC = {roc_auc:.3f})")

    ax1.plot([0, 1], [0, 1], color="grey", linestyle="--")
    ax1.set_title("A. Receiver Operating Characteristic", loc="left")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right", frameon=False)

    # PR
    for (name, probs), color in zip(
        [
            ("XGBoost", y_probs_xgb),
            ("Logistic Regression", y_probs_lr),
            ("Random Forest", y_probs_rf),
        ],
        plot_colors,
    ):
        precision, recall, _ = precision_recall_curve(y_test, probs)
        avg_prec = average_precision_score(y_test, probs)
        ax2.plot(recall, precision, color=color, label=f"{name} (AP = {avg_prec:.3f})")

    ax2.set_title("B. Precision-Recall Curve", loc="left")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="lower left", frameon=False)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}Figure2_Performance_Curves.png")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}Figure2_Performance_Curves.png")


# ==========================================
# 4. Figure 3: CV Stability
# ==========================================
def plot_cv_stability(cv_path):
    print("Generating Figure 3 (CV Stability)...")
    if not os.path.exists(cv_path):
        print(f"ERROR: {cv_path} not found.")
        return

    cv_df = pd.read_csv(cv_path)
    possible_cols = ["ROC AUC", "Fold_AUC", "AUC", "auc", "ROC_AUC"]
    y_col = next((c for c in possible_cols if c in cv_df.columns), None)

    if not y_col:
        print(f"ERROR: Could not find AUC column in {cv_path}.")
        return

    cv_df["Model"] = cv_df["Model"].str.replace("_", " ")

    model_order = [
        "RF Uncalibrated",
        "RF Calibrated",
        "LR Uncalibrated",
        "LR Calibrated",
        "XGB Uncalibrated",
        "XGB Calibrated",
    ]
    actual_order = [m for m in model_order if m in cv_df["Model"].unique()]
    plt.figure(figsize=(7.5, 4.5))

    sns.boxplot(
        x="Model",
        y=y_col,
        hue="Model",
        data=cv_df,
        order=actual_order,
        palette=okabe_ito_palette[:6],
        showfliers=False,
        width=0.6,
        boxprops=dict(alpha=0.8, edgecolor="black", linewidth=1.0),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        legend=False,
    )
    sns.swarmplot(
        x="Model",
        y=y_col,
        data=cv_df,
        order=actual_order,
        color="black",
        alpha=0.6,
        size=4,
    )

    sns.despine(trim=True, offset=5)
    plt.ylabel("ROC AUC Score")
    plt.xlabel("Model Configuration")
    plt.title("Cross-Validation Stability: Uncalibrated vs. Calibrated Models", pad=15)

    y_min = cv_df[y_col].min()
    plt.ylim(max(0, y_min - 0.05), 1.05)

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}Figure3_CV_Stability.png")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}Figure3_CV_Stability.png")


# ==========================================
# 5. Figure 5: SHAP Summary
# ==========================================
def plot_shap_summary(model, X_train, feature_names):
    print("Generating Figure 5 (Horizontal SHAP Summary)...")

    if hasattr(model, "calibrated_classifiers_"):
        booster = model.calibrated_classifiers_[0].estimator.get_booster()
    else:
        booster = model.get_booster()

    dmatrix = xgb.DMatrix(X_train, feature_names=feature_names)
    shap_matrix = booster.predict(dmatrix, pred_contribs=True)
    shap_values = shap_matrix[:, :-1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    max_display = 15
    top_indices = np.argsort(mean_abs_shap)[::-1][:max_display]
    top_features = [feature_names[i] for i in top_indices]

    plt.figure(figsize=(7.5, 3.5))
    ax = plt.gca()
    import matplotlib.colors as mcolors

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "shap_cmap", ["#1E88E5", "#FF0D57"]
    )

    for i, idx in enumerate(top_indices):
        f_shap = shap_values[:, idx]

        if isinstance(X_train, pd.DataFrame):
            f_vals = X_train.iloc[:, idx].values
        else:
            f_vals = X_train[:, idx]

        f_min, f_max = np.nanmin(f_vals), np.nanmax(f_vals)
        if f_max > f_min:
            f_norm = (f_vals - f_min) / (f_max - f_min)
        else:
            f_norm = np.zeros_like(f_vals)

        jitter = np.random.uniform(-0.25, 0.25, size=len(f_shap))
        x_pos = i + jitter

        sc = ax.scatter(
            x_pos, f_shap, c=f_norm, cmap=cmap, s=6, alpha=0.7, edgecolors="none"
        )

    ax.axhline(0, color="gray", linestyle="-", linewidth=1.0, alpha=0.5)
    ax.set_xticks(range(max_display))
    ax.set_xticklabels(top_features, rotation=45, ha="right")
    ax.set_ylabel("SHAP value\n(impact on model output)")

    cbar = plt.colorbar(sc, ax=ax, pad=0.01, aspect=30)
    cbar.set_label("Feature value")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])

    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}Figure5_SHAP_Summary.png")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}Figure5_SHAP_Summary.png")


# ==========================================
# 6. Figure 7: Combined Importance & Failure Heatmap
# ==========================================
def plot_failure_heatmap(model, X_test, y_test, df_test_meta):
    print("Generating Figure 7 (Failure Heatmap)...")
    y_pred = model.predict(X_test)

    failures_idx = np.where(y_test != y_pred)[0]
    if len(failures_idx) == 0:
        print("No failures found! Skipping heatmap.")
        return

    X_failures = X_test.iloc[failures_idx]

    if hasattr(model, "calibrated_classifiers_"):
        booster = model.calibrated_classifiers_[0].estimator.get_booster()
    else:
        booster = model.get_booster()

    dmatrix = xgb.DMatrix(X_failures, feature_names=X_test.columns.tolist())
    shap_matrix = booster.predict(dmatrix, pred_contribs=True)
    shap_values_fail = shap_matrix[:, :-1]

    mean_abs_shap = np.abs(shap_values_fail).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:15]
    top_features = X_test.columns[top_indices]

    plot_df = pd.DataFrame(shap_values_fail[:, top_indices], columns=top_features)
    failures_meta = []
    for rel_idx, i in enumerate(failures_idx):
        meta_row = df_test_meta.iloc[rel_idx]
        pred_l = y_pred[i]
        type_err = "FP" if pred_l == 1 else "FN"
        serovar = meta_row.get("Serovar", "Unknown")
        failures_meta.append(
            {"Error": type_err, "Serovar": serovar, "Original_Idx": rel_idx}
        )

    meta_df = pd.DataFrame(failures_meta)

    meta_df.sort_values(by=["Error", "Serovar"], inplace=True)
    plot_df = plot_df.iloc[meta_df.index]

    err_palette = {"FP": "#D55E00", "FN": "#0072B2"}
    unique_serovars = list(meta_df["Serovar"].unique())
    serovar_colors = sns.color_palette("tab20", n_colors=max(2, len(unique_serovars)))
    serovar_map = {
        s: serovar_colors[i % len(serovar_colors)]
        for i, s in enumerate(unique_serovars)
    }

    row_colors = pd.DataFrame(
        {
            "Error": meta_df["Error"].map(err_palette),
            "Serovar": meta_df["Serovar"].map(serovar_map),
        },
        index=plot_df.index,
    )

    max_rows = 50
    if len(plot_df) > max_rows:
        plot_df = plot_df.iloc[:max_rows]
        row_colors = row_colors.iloc[:max_rows]

    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    g = sns.clustermap(
        plot_df,
        row_cluster=False,
        col_cluster=False,
        row_colors=row_colors,
        cmap=cmap,
        center=0,
        figsize=(7.5, max(5, len(plot_df) * 0.2)),
        linewidths=0.5,
        dendrogram_ratio=0.01,
    )

    g.cax.set_visible(False)

    g.ax_row_colors.set_xticks([0.5, 1.5])
    g.ax_row_colors.set_xticklabels(
        ["Error", "Serovar"], rotation=45, ha="right", fontsize=10
    )

    ax = g.ax_heatmap
    ax.set_xticks(np.arange(plot_df.shape[1]) + 0.5)
    ax.set_xticklabels(
        top_features, rotation=45, ha="right", rotation_mode="anchor", fontsize=10
    )

    plt.setp(ax.get_yticklabels(), visible=False)

    g.fig.suptitle("Forensic Analysis of Misclassifications", y=1.02)

    from matplotlib.patches import Patch

    err_patches = [Patch(facecolor=err_palette[k], label=k) for k in err_palette]
    ser_patches = [
        Patch(facecolor=serovar_map[s], label=s) for s in unique_serovars[:6]
    ]

    g.ax_heatmap.legend(
        handles=err_patches + ser_patches,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
    )

    g.savefig(f"{OUTPUT_DIR}Figure7_Failure_Heatmap.png", bbox_inches="tight")
    plt.close(g.fig)
    print(f"Saved: {OUTPUT_DIR}Figure7_Failure_Heatmap.png")


# ==========================================
# Export for R
# ==========================================
def export_for_R_oncoprint(df):
    print("Exporting data for R (Figure 4)...")
    top_genes = [
        "qnrB19",
        "gyrA_D87Y",
        "blaTEM-1",
        "tet(A)",
        "fosA7",
        "aac(3)-IIe",
        "blaCTX-M-15",
        "aadA1",
        "sul2",
    ]
    valid_genes = [g for g in top_genes if g in df.columns]

    if valid_genes:
        df[["Assembly", "Serovar", "label"]].to_csv(
            f"{OUTPUT_DIR}metadata_for_heatmap.csv", index=False
        )
        df.set_index("Assembly")[valid_genes].T.to_csv(
            f"{OUTPUT_DIR}top_genes_matrix.csv"
        )
        print("Export complete.")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df = pd.read_csv(DATA_PATH)

    generate_table1(df)
    plot_serovar_treemap(df)
    plot_cv_stability(MODEL_COMPARISON_PATH)

    model, X_train, X_test, y_train, y_test, feats, comp_models = (
        get_live_model_and_data(df)
    )

    df_meta = df.loc[X_test.index, ["Assembly", "Serovar", "label"]]

    plot_performance_curves(comp_models, X_test, y_test)
    plot_shap_summary(model, X_train, feats)
    plot_failure_heatmap(model, X_test, y_test, df_meta)

    export_for_R_oncoprint(df)
