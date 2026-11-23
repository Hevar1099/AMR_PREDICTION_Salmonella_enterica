import os
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import StratifiedTable

os.makedirs("Output", exist_ok=True)

df = pd.read_csv("Data/final_ml_dataset.csv")
col = "aph(3'')-Ib"

rows = []


def add_fisher_row(name, sub):
    a = int(((sub[col] == 1) & (sub["label"] == 1)).sum())  # gene+, R
    b = int(((sub[col] == 1) & (sub["label"] == 0)).sum())  # gene+, S
    c = int(((sub[col] == 0) & (sub["label"] == 1)).sum())  # gene-, R
    d = int(((sub[col] == 0) & (sub["label"] == 0)).sum())  # gene-, S
    odds, pval = fisher_exact([[a, b], [c, d]])
    r_pos = a / (a + b) * 100 if (a + b) > 0 else np.nan
    r_neg = c / (c + d) * 100 if (c + d) > 0 else np.nan
    rows.append(
        {
            "Scope": name,
            "N": len(sub),
            "Gene_Pos_R": a,
            "Gene_Pos_S": b,
            "Gene_Neg_R": c,
            "Gene_Neg_S": d,
            "R_Pct_Gene_Pos": round(r_pos, 2) if not np.isnan(r_pos) else np.nan,
            "R_Pct_Gene_Neg": round(r_neg, 2) if not np.isnan(r_neg) else np.nan,
            "Odds_Ratio": round(float(odds), 4) if np.isfinite(odds) else np.inf,
            "P_Value": float(f"{pval:.4g}"),
        }
    )


add_fisher_row("Overall_Unadjusted", df)

for ser in ["Infantis", "Enteritidis", "Monophasic Variant", "Kentucky", "Saintpaul", "Senftenberg"]:
    sub = df[df["Serovar"] == ser]
    add_fisher_row(f"Within_{ser}", sub)

# Mantel-Haenszel stratified OR controlling for Serovar (all two-class serovars)
tables = []
for ser, sub in df.groupby("Serovar"):
    a = int(((sub[col] == 1) & (sub["label"] == 1)).sum())
    b = int(((sub[col] == 1) & (sub["label"] == 0)).sum())
    c = int(((sub[col] == 0) & (sub["label"] == 1)).sum())
    d = int(((sub[col] == 0) & (sub["label"] == 0)).sum())
    if (a + c) > 0 and (b + d) > 0 and (a + b) > 0 and (c + d) > 0:
        tables.append(np.array([[a, b], [c, d]]))

st = StratifiedTable(tables)
mh_or = float(st.oddsratio_pooled)
mh_ci_lo, mh_ci_hi = st.oddsratio_pooled_confint()
mh_pval = float(st.test_null_odds().pvalue)

# Also stratify by (Serovar, gyrA_D87Y)
tables_gyra = []
for _, sub in df.groupby(["Serovar", "gyrA_D87Y"]):
    a = int(((sub[col] == 1) & (sub["label"] == 1)).sum())
    b = int(((sub[col] == 1) & (sub["label"] == 0)).sum())
    c = int(((sub[col] == 0) & (sub["label"] == 1)).sum())
    d = int(((sub[col] == 0) & (sub["label"] == 0)).sum())
    if (a + c) > 0 and (b + d) > 0 and (a + b) > 0 and (c + d) > 0:
        tables_gyra.append(np.array([[a, b], [c, d]]))

st_g = StratifiedTable(tables_gyra)
mhg_or = float(st_g.oddsratio_pooled)
mhg_ci_lo, mhg_ci_hi = st_g.oddsratio_pooled_confint()
mhg_pval = float(st_g.test_null_odds().pvalue)

res_df = pd.DataFrame(rows)
res_df.to_csv("Output/aph3_adjusted_results.csv", index=False)

print(res_df.to_string(index=False))
print(f"\nMantel-Haenszel OR (adjusted for Serovar, {len(tables)} informative strata): "
      f"OR={mh_or:.4f} (95% CI: {mh_ci_lo:.4f}-{mh_ci_hi:.4f}), p={mh_pval:.4g}")
print(f"Mantel-Haenszel OR (adjusted for Serovar + gyrA_D87Y, {len(tables_gyra)} informative strata): "
      f"OR={mhg_or:.4f} (95% CI: {mhg_ci_lo:.4f}-{mhg_ci_hi:.4f}), p={mhg_pval:.4g}")
