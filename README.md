# Laboratory Measurement Connection Discovery via Advanced Statistics and Explainable AI

[![All of Us Workspace](https://img.shields.io/badge/All_of_Us-Workbench_Workspace-blue.svg)](https://workbench.verily.com/workspaces/connection-discovery)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](https://www.python.org/)

> **Repository for code developed in the *All of Us* Researcher Workbench.**  
> 🔗 **Workspace Link:** [All of Us Workbench - Connection Discovery Workspace](https://workbench.verily.com/workspaces/connection-discovery)

This project investigates whether advanced non-monotonic statistical methods and Explainable AI (XAI) feature attribution techniques can discover novel, complex biological relationships between continuous laboratory measurements that traditional linear correlation methods fail to detect.

---

## 📌 Table of Contents

- [Research Overview](#-research-overview)
- [Repository Structure](#-repository-structure)
- [Methodological Framework](#-methodological-framework)
  - [Statistical Correlation Metrics](#1-statistical-correlation-metrics)
  - [Explainable AI & Machine Learning](#2-explainable-ai--machine-learning)
- [Data Processing Pipeline](#-data-processing-pipeline)
- [Experimental Design](#-experimental-design)
- [Environment & Setup](#-environment--setup)
- [Execution Order](#-execution-order)
- [Data Privacy & Compliance](#-data-privacy--compliance)

---

## 🔬 Research Overview

Standard clinical analysis heavily relies on linear ($r$) or monotonic ($\rho$) correlation frameworks to understand biomarker associations. However, human biological systems exhibit complex, non-linear, state-dependent, and multi-phasic dependencies.

Using **HbA1c** and **Insulin** as initial anchor target analytes, this project maps systemic, non-linear connections across broader laboratory test panels within the *All of Us* Electronic Health Record (EHR) database.

---

## 📁 Repository Structure

```text
AoU_connection_discovery/
│
├── results/                                # Output directory for partial and processed result files
├── .gitignore                            # Git exclusion rules
├── Helper_Functions.py                     # Central utility library (unit conversion, stats, evaluations)
├── lior_XAI_env.yaml                       # Conda environment definition
│
├── get single lab data with cleaning.ipynb  # Step 1: Regex extraction of anchor labs & co-occurring lab retrieval
├── columns semantic clustering.ipynb        # Step 2: SapBERT/MedEmbed embedding & agglomerative clustering
├── data cleaning 1.ipynb                    # Step 3: Distribution verification, unit normalization & entropy filtering
├── Correlations tests.ipynb                 # Step 4: Non-monotonic metric execution & GBDT/XAI model training
└── look at correlations results.ipynb       # Step 5: Result analysis, consensus evaluation & visualizations
```

---

## 📐 Methodological Framework

### 1. Statistical Correlation Metrics

We assess continuous lab associations across three levels of dependency:

- **Linear:** Pearson Correlation ($r$)
- **Monotonic:**
  - Spearman Rank Correlation ($\rho$)
  - Kendall rank correlation (τ)
- **Non-Monotonic / Complex:**
  - Distance Correlation ($\text{dCor}$)
  - Maximal Information Coefficient ($\text{MIC}$)
  - Chatterjee’s Rank Correlation ($\xi$)
  - Hoeffding’s $D$

### 2. Explainable AI & Machine Learning

To benchmark non-linear feature importance against statistical metrics, we utilize:

- **Gradient Boosted Decision Trees:**
  - **XGBoost:** Weight, Gain, and Cover Importance
  - **CatBoost:** PredictionValuesChange
  - **LightGBM:** Split and Gain Importance
- **Model-Agnostic Explainability:**
  - **SHAP (SHapley Additive exPlanations):** TreeExplainer attribution values

---

## 🛠 Data Processing Pipeline

```
       [ All of Us EHR Data ]
                 │
                 ▼
 ┌──────────────────────────────┐
 │ 1. Anchor Extraction & Fetch │  (Regex matching for HbA1c & Insulin co-occurrences)
 └──────────────┬───────────────┘
                │
                ▼
 ┌──────────────────────────────┐
 │ 2. Unit Normalization        │  (Distributional empirical alignment per lab test)
 └──────────────┬───────────────┘
                │
                ▼
 ┌──────────────────────────────┐
 │ 3. Reshaping & Aggregation   │  (Long-to-wide transformation per person/visit date)
 └──────────────┬───────────────┘
                │
                ▼
 ┌──────────────────────────────┐
 │ 4. Semantic Harmonization    │  (SapBERT / MedEmbed + Agglomerative Clustering)
 └──────────────┬───────────────┘
                │
                ▼
 ┌──────────────────────────────┐
 │ 5. Variance Filtering        │  (Perplexity & entropy variance thresholding)
 └──────────────────────────────┘
```

---

## 🧪 Experimental Design

### **Experiment 1: The Correlation Hierarchy**
* **Goal:** Quantify the yield of novel biological relationships discovered exclusively through non-monotonic metrics ($\text{dCor}$) vs. linear ($r$) and monotonic ($\rho$ + τ) metrics.

### **Experiment 2: Multi-Method Consensus Evaluation**
* **Goal:** Evaluate cross-method agreement across non-monotonic statistical metrics ($\text{MIC}$, $\xi$, $D$) and tree-based XAI metrics (XGBoost, CatBoost, LightGBM, SHAP) to isolate high-confidence non-linear associations.

### **Experiment 3: Feature Selection for Machine Learning**
* **Goal:** Test statistical correlation metrics as upstream feature selection filters for predictive models across 4 regimes:
  1. **Linear Only** ($r$)
  2. **Linear + Monotonic** ($r + \rho$)
  3. **Statistically Significant Set** (Any metric passing threshold)
  4. **Unfiltered Set** (Complete feature space)

---

## ⚙️ Environment & Setup

To replicate the execution environment locally or within a Jupyter instance:

# Clone the repository
add [https://github.com/lior0110/AoU_connection_discovery.git](https://github.com/lior0110/AoU_connection_discovery.git) to Git repositories in apps section

# Create and activate conda environment
conda env create -f lior_XAI_env.yaml
conda activate lior_XAI_env
```

---

## 🚀 Execution Order

To run the complete pipeline sequentially:

1. **`get single lab data with cleaning.ipynb`**: Extract anchor lab co-occurrences.
2. **`columns semantic clustering.ipynb`**: Cluster synonymous test names using medical embeddings.
3. **`data cleaning 1.ipynb`**: Perform unit harmonization, pivot data, and apply quality filters.
4. **`Correlations tests.ipynb`**: Calculate linear, monotonic, and non-monotonic statistics alongside GBDT model training.
5. **`look at correlations results.ipynb`**: Analyze results, evaluate consensus, and generate comparative visualizations.

---

## 🔐 Data Privacy & Compliance

This workspace code interacts with the **All of Us Research Program** EHR dataset.
- All individual-level data resides within the secure *All of Us* Researcher Workbench environment.
- No individual-level data, Protected Health Information (PHI), or Personally Identifiable Information (PII) is exported or included in this repository.
- All reported outputs conform to the *All of Us* Data User Code of Conduct guidelines regarding cell counts and aggregation.
