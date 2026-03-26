# Antimicrobial Resistance (AMR) Prediction via Genomic Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.0%2B-blue.svg)](https://www.r-project.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.1.3-orange.svg)](https://xgboost.readthedocs.io/)
[![KMC3](https://img.shields.io/badge/KMC-3.2.4-green.svg)](https://github.com/refresh-bio/KMC)

## Project Overview
This repository contains a comprehensive, highly optimized machine learning pipeline for predicting Antimicrobial Resistance (AMR)—specifically Ciprofloxacin resistance—from bacterial genomic data. 
To ensure rigorous clinical relevance and prevent lineage-memorization, the pipeline features:

* **Dual Feature Modalities:** Supports both curated AMR gene presence/absence matrices and highly parallelized, hashed *k*-mer frequencies.
* **Lineage-Aware Filtering:** Statistically isolates and removes highly correlated lineage markers that do not drive true resistance.
* **Clinical Evaluation Metrics:** Evaluates models using standard ML metrics alongside clinical microbiology standards (Very Major Errors, Major Errors, Isotonic Calibration, and Decision Curve Analysis).
* **Interpretability & Visualization:** Utilizes SHAP values for forensic failure analysis and incorporates advanced R-based visualizations (ComplexHeatmap) to map predictive features to clinical phenotypes.

---

## Repository Structure

```text
├── Data/                          
│   ├── accessions.txt
│   ├── accessions_list.txt
│   ├── accessions_clean.txt       # Use this to fetch the NCBI genomes
│   ├── cleaned_metadata.csv
│   ├── dna_file_paths.csv
│   ├── final_ml_dataset.csv
│   ├── metadata.tsv
│   ├── test_set.csv
│   ├── train_set.csv
│   ├── X_kmer_matrix.npz
│   └── y_kmer_labels.csv
├── Documentation/                 # Supplementary project documentation
├── Scripts/                       # Core pipeline scripts (Python & R)
│   ├── main_script.py
│   ├── Cleaning_genome.py
│   ├── file_kmers.py
│   ├── Kmers_counting.py
│   ├── kmc_worker.py
│   ├── hashing_collision.py
│   ├── train_kmer_model.py
│   ├── ML_testing.py
│   ├── serovar_metrics.py
│   ├── Cross_over_validation.py
│   ├── data_viz_python.py
│   └── data_viz_R.R
├── Output/                        # Generated models, metrics, and publication figures
│   ├── best_lr_model.pkl
│   ├── best_rf_model.pkl
│   ├── best_xgb_model.pkl
│   ├── feature_importance.png
│   ├── Figure1_Serovar_Treemap.png
│   ├── Figure2_Performance_Curves.png
│   ├── Figure3_CV_Stability.png
│   ├── Figure4_OncoPrint.pdf
│   ├── Figure5_SHAP_Summary.png
│   ├── Figure7_Failure_Heatmap.png
│   ├── Figure8_Calibration.png
│   ├── Figure9_Stability_Decay.png
│   ├── Figure11_DCA.png
│   ├── loso_performance.png
│   ├── loso_results.csv
│   ├── metadata_for_heatmap.csv
│   ├── model_calibration_comparison.csv
│   ├── model_comparison.csv
│   ├── serovar_specific_metrics.csv
│   ├── Table1_demographics.csv
│   ├── test_indices.npy
│   └── top_genes_matrix.csv
├── environment.yml                # Conda environment configuration
├── requirements.txt               # Standard Python dependencies
└── README.md
```

---

## Prerequisites & Setup

This project uses a mix of Python, R, and external bioinformatics tools (KMC3). The easiest and most reproducible way to set up the environment is using the provided Conda `environment.yml` file.

### Option 1: Conda Environment (Recommended)
This will install Python, R, all required libraries, and KMC3 in one go.
```bash
conda env create -f environment.yml
conda activate amr_project
```

### Option 2: Standard Pip & R Installation
If you prefer not to use Conda for the entire environment, you can install the dependencies manually:

1. **Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **KMC3:** Must be installed and accessible in your system's PATH.
   ```bash
   conda install -c bioconda kmc
   ```
3. **R Dependencies:** Open R and run:
   ```R
   install.packages("circlize")
   if (!requireNamespace("BiocManager", quietly = TRUE))
       install.packages("BiocManager")
   BiocManager::install("ComplexHeatmap")
   ```

---

## Data Retrieval Protocol
To adhere to GitHub file size limits and ensure reproducibility, raw genome sequences are not hosted in this repository. 

1. Run `Scripts/main_script.py` (Phase 1) to generate the clean accession lists.
2. Use the NCBI Datasets CLI to download the raw bacterial assemblies:
   ```bash
   datasets download genome accession --inputfile Data/accessions_clean.txt --filename genomes.zip
   unzip genomes.zip -d Data/Genome/ncbi_dataset/data/
   ```

---

## Execution Pipeline

### Phase 1: Data Preparation & Gene Feature Engineering
**1. Metadata Preprocessing (`main_script.py`)**
* Cleans raw NCBI metadata (`metadata.tsv`), calculates antibiotic frequencies, isolates Ciprofloxacin phenotypes (R/S), and one-hot encodes AMR genotypes.

**2. Lineage Filtering (`Cleaning_genome.py`)**
* Standardizes serovar nomenclature and performs strict feature selection by dropping low-variance genes and lineage markers (>95% prevalence) lacking true resistance correlation.

### Phase 2: Agnostic *K*-mer Extraction
**3. Genome Mapping (`file_kmers.py`)**
* Scans the downloaded NCBI genomes and maps `.fna` file paths to target Assembly IDs.

**4. High-Performance *K*-mer Extraction (`Kmers_counting.py` & `kmc_worker.py`)**
* Dynamically spins up CPU workers to run KMC3. Uses the MurmurHash3 "Hashing Trick" to map 31-mers to a fixed-size (16.7M) memory-efficient SciPy CSR sparse matrix.

**5. Hashing Vector Audit (`hashing_collision.py`)**
* A theoretical quality-control script to calculate the collision probability of the feature hashing trick based on unique dataset *k*-mers.

### Phase 3: Model Training & Stability Auditing
**6. *K*-mer Stability Auditing (`train_kmer_model.py`)**
* Uses L1-penalized Stability Selection to isolate the most robust *k*-mers. Compares feature stability decay directly against the gene model and trains an ElasticNet classifier evaluated on clinical VME limits. Outputs `Figure9_Stability_Decay.png`.

**7. Gene Model Training & Rigorous Testing (`ML_testing.py`)**
* Tunes XGBoost, Random Forest, and Logistic Regression on curated AMR genes. Performs Isotonic Calibration and evaluates standard (ROC AUC) and clinical metrics (Brier Score, ME%, VME%). Performs Blocked Cross-Validation.

### Phase 4: Clinical Validation & Out-of-Lineage Generalization
**8. Serovar-Specific Granularity (`serovar_metrics.py`)**
* Evaluates the calibrated model's performance on high-interest clinical lineages (*Enteritidis*, *Heidelberg*, etc.) utilizing strict test-set indices (`test_indices.npy`).

**9. Out-of-Lineage Generalization (`Cross_over_validation.py`)**
* Performs Leave-One-Serovar-Out (LOSO) cross-validation and calculates genetic Jaccard distances to prove the model identifies core resistance mechanisms rather than memorizing lineage patterns.

### Phase 5: Visualization & Interpretability
**10. Publication Figures & SHAP Analysis (`data_viz_python.py`)**
* Generates publication-ready demographics tables, Treemaps, and ROC/PR curves. Features an advanced "Forensic Failure Heatmap" clustering misclassified samples based on their SHAP profiles. Exports data for R plotting.

**11. Complex Heatmap Generation (`data_viz_R.R`)**
* Reads the exported python matrices into R to generate a publication-ready annotated `ComplexHeatmap` (`Figure4_OncoPrint.pdf`), applying hierarchical clustering to map phenotypic resistance against predictive AMR features.