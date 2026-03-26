# Carer Segmentation Analysis

A data-driven segmentation of 11,767 UK carer assessments from Birmingham, identifying distinct carer types and their wellbeing needs.

## Project Overview

Unpaid carers are not a homogeneous group. This project uses clustering analysis to identify **five distinct caring situations** and **four patterns of wellbeing need**, revealing that knowing someone's caring situation alone doesn't predict how they're coping — a key insight for designing targeted support services.


## Notebooks

### `src/01_data_cleaning.ipynb`
- Fixes encoding corruption across text fields
- Parses multi-select disability fields into 9 standardised condition categories (cognitive/neurological, neurodevelopmental, mental health, musculoskeletal, metabolic, cardiorespiratory, sensory, cancer, general chronic)
- Splits composite score fields (e.g. `"3 - Sometimes"`) into numeric and label columns
- Resolves data integrity issues: underage carers, impossible hours, swapped ages, implausible values, duplicates
- Outputs `carers_cleaned.csv`

### `src/02_clustering_analysis.ipynb`
- **Feature selection**: Separates features into situation variables (caring role), need variables (wellbeing scores), and profiling variables (demographics, held out)
- **Imputation & scaling**: Column-specific strategies — KNN for correlated scores, median for skewed continuous, zero for binary flags. Robust scaling for skewed variables, min-max for bounded scores.
- **Kernel transformation**: Nyström RBF kernel approximation to capture non-linear feature interactions
- **Clustering**: Compares KMeans, Spectral, and Hierarchical (Ward) across k=3–7. Selects KMeans k=5 (situation) and k=4 (needs) based on Davies-Bouldin, Calinski-Harabasz, cluster balance, and bootstrap stability
- **GMM analysis**: Gaussian Mixture Models with BIC/AIC model selection and soft membership entropy analysis
- **Profiling & visualisation**: Cluster profiles, cross-tabulation, and all report figures

### `src/03_supervised_analysis.ipynb`

#### Phase 3A: Binary Outcome Prediction

- **Target engineering:** Binarises 9 Likert-scored domains — Low (1–2) vs High (4–5), score 3 dropped as ambiguous
- **Intervention extraction:** Rule-based keyword matching on free-text `summary_actions` to extract 10 binary intervention flags (CERS referral, benefits advice, wellbeing payment, respite, etc.)
- **Model:** HistGradientBoostingClassifier with regularisation (max_depth=3, L2=1.0, min_samples_leaf=50, early stopping)
- **Evaluation:** 5-fold stratified CV, threshold tuning on train folds only. ROC–AUC 0.64–0.80 across domains. Strongest: Financial Situation (0.80), Caring Commitments (0.78), Work/Education (0.74)
- **Permutation importance:** Model-agnostic, computed on held-out folds (10 repeats). Top drivers: working status, caring hours, carer age, condition complexity
- **Category-level effects:** Marginal effects on P(High) per categorical feature level, filtered to n ≥ 50
- **Outputs:** `binary_prediction_summary.csv`, `binary_perm_importance_<domain>.csv`, `category_effects_<domain>.csv`, Figure 1 (performance overview), Figure 2 (top predictors), per-domain ROC/PR/confusion matrix plots

#### Phase 3B: Intervention Impact Modelling

- **Data:** 2,380 carers with 6-month review. Improvement target: review − baseline ≥ 1
- **Model:** Logistic Regression (balanced class weights) — chosen for interpretability over gradient boosting
- **Marginal effects:** For each intervention × domain, toggles intervention ON vs OFF across reference population. Reports Δ P(improved) adjusted for baseline features and score
- **Caveat:** Adjusted associations, not causal. Interventions were not randomly assigned. Results are a prioritisation tool, not a prescription
- **Key findings:** Wellbeing payments positive across multiple domains. CERS referrals linked to safety improvement. Benefits advice linked to financial improvement. No actions flag linked to lower improvement
- **Outputs:** `improvement_model_summary.csv`, `improve_importance_<domain>.csv`, `improve_interv_importance_<domain>.csv`, Figure 3 (intervention effect heatmap with prevalence)

### `src/04_clustering_analysis.ipynb`
•⁠  ⁠*Inputs:* ⁠ clustering_kmeans1.csv ⁠ (output of ⁠ 02_clustering_analysis.ipynb ⁠), ⁠ Data_Review_Cleaned.csv ⁠, and an interventions dataframe with columns ⁠ lookup ⁠ (carer ID) and ⁠ intervention_category ⁠
•⁠  ⁠Links initial assessments to follow-up review data (~2,000 of 11,800 carers matched by ID)
•⁠  ⁠Computes completion rates by situation cluster and estimates the opportunity gap: how many additional completions each cluster would yield if it matched the best-performing group's rate
•⁠  ⁠Calculates domain-level change scores across nine wellbeing areas (review − initial; negative = improvement)
•⁠  ⁠Computes Cohen's d effect sizes and paired t-tests for each cluster × domain combination, categorised as small / moderate / strong
•⁠  ⁠Merges intervention records and produces two intervention heatmaps: score change by intervention type × domain, and overall score change by intervention type × situation cluster
•⁠  ⁠Outputs five figures: completion diagnostic, improvement rate by cluster (95% CIs), effect size heatmap, intervention-domain heatmap, and intervention-cluster heatmap

## Key Findings

| Situation Type | % | Description |
|---------------|---|-------------|
| Caring for Complex Parents | 22% | Working-age adults caring for parents with 3.5 conditions avg |
| The Broad Middle | 34% | Typical carers, straightforward situations, 45 hrs/week |
| Round-the-Clock Carers | 13% | 131 hrs/week, 60% live with the person |
| Retired Spousal Carers | 13% | Avg age 78, 85% retired, financially stable |
| Lifelong Parent Carers | 17% | 25 years avg, caring for adult children |

| Needs Pattern | % | Key Struggle |
|--------------|---|-------------|
| Struggling Everywhere | 24% | Crisis across all dimensions |
| Health & Time Squeezed | 28% | Own health declining, no time for themselves |
| Work & Finance Squeezed | 25% | Can't balance caring with earning |
| Coping Well | 23% | Managing across the board |

**Central insight:** Situation does not determine needs. Each situation type contains all four needs patterns. Support services must assess both independently.

## Technical Details

- **Clustering method**: KMeans on Nyström RBF kernel features
- **Bootstrap stability**: ARI = 0.978 ± 0.007
- **Validation**: Davies-Bouldin Index, Calinski-Harabasz Index, hierarchical dendrograms, GMM comparison
- **Data**: ~11,800 carer assessments, Birmingham UK, 2018–2022

## Requirements

- pandas
- numpy 
- scikit-learn 
- scipy 
- matplotlib 
- seaborn


## Usage

```bash
# Run cleaning first
jupyter notebook src/01_data_cleaning.ipynb

# Then analysis
jupyter notebook src/02_clustering_analysis.ipynb