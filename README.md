# Beyond the Black Box: Ciprofloxacin Resistance Prediction in *Salmonella enterica*

[![Python](https://img.shields.io/badge/Python-3.10.19-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.0%2B-blue.svg)](https://www.r-project.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.1.3-orange.svg)](https://xgboost.readthedocs.io/)
[![KMC3](https://img.shields.io/badge/KMC-3.2.4-green.svg)](https://github.com/refresh-bio/KMC)

Code, curated datasets, trained models, and audit outputs for *"Beyond the Black Box: A Calibrated, Gene-Centric Pipeline for High-Precision Ciprofloxacin Resistance Prediction in Salmonella enterica"*. This repository compares an alignment-free 31-mer MurmurHash3 ($2^{24}$) pipeline against curated 22-gene machine learning models (Logistic Regression, Random Forest, and XGBoost) across 8,759 *Salmonella enterica* isolates (511 resistant, 8,248 susceptible) from the NCBI Pathogen Detection database, including isotonic probability calibration, SHAP interpretability, Leave-One-Serovar-Out (LOSO) validation, and population-structure sensitivity audits.

---

## Directory Structure

```text
.
├── Data/                         # Curated metadata, accession lists, and ML matrices
│   ├── accessions_clean.txt      # Final NCBI Assembly accessions (N = 8,759)
│   ├── cleaned_metadata.csv      # Cleaned metadata with standardized serovar names
│   ├── final_ml_dataset.csv      # Curated 22-gene binary feature matrix + labels
│   ├── metadata.tsv              # Raw NCBI Pathogen Detection export (retrieved Jan 17, 2026)
│   ├── train_set.csv             # Stratified 80% training split (N = 7,008)
│   ├── test_set.csv              # Stratified 20% test split (N = 1,751)
│   ├── X_kmer_matrix.npz         # Sparse hashed 31-mer feature matrix (2^24 space)
│   └── y_kmer_labels.csv         # Binary labels aligned to X_kmer_matrix.npz
├── Script/                       # Core pipeline, plotting, and statistical audit scripts
├── Output/                       # Saved models (.pkl), tables (.csv), and figures (.png/.pdf)
├── environment.yml               # Conda environment specification
├── requirements.txt              # Pinned Python dependencies + R package install lines
└── README.md
```

---

## Environment & Installation

All analyses were run with **Python 3.10.19** and fixed random seed `42` using pinned package versions (`pandas==2.3.3`, `numpy==2.2.6`, `scipy==1.15.2`, `statsmodels==0.15.0`, `scikit-learn==1.7.2`, `xgboost==3.1.3`, `joblib==1.5.3`, `shap==0.48.0`, `mmh3==5.2.0`, `matplotlib==3.10.8`, `seaborn==0.13.2`, `squarify==0.4.3`). R (`ComplexHeatmap`, `circlize`) is only required to regenerate the OncoPrint figure (`Figure4_OncoPrint.pdf`), and KMC3 (`3.2.4`) is only required if re-extracting 31-mers from raw assemblies.

### Option 1: Conda (Recommended)

```bash
git clone https://github.com/Hevar1099/AMR_PREDICTION_Salmonella_enterica.git
cd AMR_PREDICTION_Salmonella_enterica
conda env create -f environment.yml
conda activate amr_project
```

### Option 2: Pip + R + KMC3

```bash
git clone https://github.com/Hevar1099/AMR_PREDICTION_Salmonella_enterica.git
cd AMR_PREDICTION_Salmonella_enterica
pip install -r requirements.txt
```

If regenerating the OncoPrint in R:

```r
install.packages("circlize")
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("ComplexHeatmap")
```

If re-extracting k-mers from raw genome assemblies:

```bash
conda install -c bioconda kmc
```

---

## Quick Start

All scripts use relative paths (`Data/...`, `Output/...`) and should be run from the repository root. To verify the environment and reproduce the primary test-set metrics and paired DeLong tests on the included dataset (`N = 1,751`; 100 resistant, 1,651 susceptible):

```bash
python Script/test_set_full_metrics.py
```

Expected outputs:
- `Output/test_set_full_metrics.csv` — Accuracy, precision, recall, specificity, F1, ROC AUC, PR-AUC (with 1,000-iteration bootstrap 95% CIs), VME%, ME%, and raw TP/FN/FP/TN counts for all uncalibrated and calibrated models.
- `Output/delong_comparisons.csv` — Pairwise DeLong test $z$-statistics and $p$-values across models.

---

## Data Provenance & Retrieval

- **Source:** [NCBI Pathogen Detection](https://www.ncbi.nlm.nih.gov/pathogens/) *Salmonella enterica* Isolates Browser export, retrieved **January 17, 2026** (`Data/metadata.tsv`, 9,857 initial records).
- **Filtering:** Excluding 299 non-binary/missing ciprofloxacin AST records (123 I, 151 NS, 24 missing, 1 ND), 201 missing assembly accessions, and 598 missing/unknown serovars yields **8,759 isolates** (511 resistant [5.83%], 8,248 susceptible [94.17%]) across 93 serovars (`Data/final_ml_dataset.csv`).
- **Raw Genome Assemblies:** Not hosted in this repository due to size limits. To download the exact assemblies used for the k-mer audit via the NCBI Datasets CLI:

```bash
datasets download genome accession --inputfile Data/accessions_clean.txt --filename genomes.zip
unzip genomes.zip -d Data/Genome/ncbi_dataset/data/
```

---

## Reproducing Paper Results

Run commands from the repository root in the order below (or run individual audit scripts directly using the pre-built `Data/final_ml_dataset.csv` and saved models in `Output/`).

### Core Pipeline

| Stage | Command | Key Outputs |
| :--- | :--- | :--- |
| 1. Metadata cleaning & phenotype extraction | `python Script/main_script.py` | `Data/cleaned_metadata.csv`, `Data/accessions_clean.txt` |
| 2. Gene feature engineering & lineage filter | `python Script/Cleaning_genome.py` | `Data/final_ml_dataset.csv` (22 features) |
| 3. Hashing collision & memory audit | `python Script/hashing_collsion.py` | Console collision, occupancy, and memory report |
| 4. K-mer stability audit (requires `X_kmer_matrix.npz`) | `python Script/agnostic_kmer_stability_audit.py` | `Output/Figure9_Stability_Decay.png` |
| 5. Model training, HPO, calibration & DCA | `python Script/ML_testing.py` | `Output/best_*_model.pkl`, `Output/Figure8_Calibration.png`, `Output/Figure11_DCA.png` |
| 6. Serovar-specific metrics (Table 5) | `python Script/serovar_metrics.py` | `Output/serovar_specific_metrics.csv` |
| 7. LOSO validation & plot (Figure 8) | `python Script/Cross_over_validation.py` | `Output/loso_results.csv`, `Output/loso_performance.png` |
| 8. Figures 1, 2, 6, 7, 10 & R export | `python Script/data_viz_python.py` | `Output/Figure1_Serovar_Treemap.png`, `Figure2_Performance_Curves.png`, `Figure3_CV_Stability.png`, `Figure5_SHAP_Summary.png`, `Figure7_Failure_Heatmap.png` |
| 9. OncoPrint (Figure 9) | `Rscript Script/data_viz_R.R` | `Output/Figure4_OncoPrint.pdf` |

---

## Validation & Sensitivity Audits

Each script below runs independently from the repository root on `Data/final_ml_dataset.csv` (and `Data/cleaned_metadata.csv` where raw annotations are audited) to verify the statistical checks reported in the manuscript:

| Audit | Command | Output File(s) |
| :--- | :--- | :--- |
| Test-set metrics, bootstrap 95% CIs & DeLong tests | `python Script/test_set_full_metrics.py` | `Output/test_set_full_metrics.csv`, `Output/delong_comparisons.csv` |
| Calibration-in-the-large, slope & bin counts | `python Script/calibration_metrics.py` | `Output/calibration_diagnostics.csv`, `Output/calibration_bin_counts.csv` |
| Genotype-clustered & serovar-grouped 5-fold CV | `python Script/grouped_split_validation.py` | `Output/grouped_split_results.csv` |
| Within-serovar 5-fold & genotype-grouped CV | `python Script/within_serovar_validation.py` | `Output/within_serovar_results.csv` |
| SHAP stability, gain vs. SHAP & raw QRDR audit | `python Script/shap_stability_check.py` | `Output/shap_stability_summary.csv`, `Output/raw_quinolone_determinants.csv` |
| Serovar- & `gyrA_D87Y`-adjusted `aph(3'')-Ib` analysis | `python Script/aph3_adjusted_analysis.py` | `Output/aph3_adjusted_results.csv` |
| 5-fold GroupKFold composition & Fold 1 outlier audit | `python Script/cv_fold_audit.py` | `Output/cv_fold_composition.csv`, `Output/cv_model_summary.csv` |
| Full 93-serovar LOSO audit, CIs & AUC < 0.5 diagnostics | `python Script/loso_full_audit.py` | `Output/loso_full_audit.csv`, `Output/loso_low_auc_diagnostics.csv` |

---

## Citation

If you use this code or curated dataset, please cite:

```bibtex
@article{barznji2026beyond,
  author  = {Barznji, Hevar N.},
  title   = {Beyond the Black Box: A Calibrated, Gene-Centric Pipeline for High-Precision Ciprofloxacin Resistance Prediction in Salmonella enterica},
  journal = {Journal of Microbiological Methods},
  year    = {2026},
  note    = {Manuscript under revision},
  url     = {https://github.com/Hevar1099/AMR_PREDICTION_Salmonella_enterica}
}
```

---

## Funding, Issues & Contact

- **Author & Maintainer:** Hevar N. Barznji, Medical Laboratory Science Department, Charmo University, Sulaymaniyah, Iraq (`hivar.neyaz@chu.edu.iq`).
- **Funding:** This research received no external funding from public, commercial, or not-for-profit agencies.
- **Issues:** Please report bugs or reproducibility questions via [GitHub Issues](https://github.com/Hevar1099/AMR_PREDICTION_Salmonella_enterica/issues).
- **Data & Code Reuse:** Underlying isolate metadata are public domain via NCBI Pathogen Detection. Please contact the author regarding reuse permissions if a license file has not yet been added to this repository.
