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