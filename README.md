# Interpretable machine learning for the ankle–brachial index in type 2 diabetes

Analysis code for:

> Yang C-C, Chang C-C, Chu T-W, Pei D, Lin C-Y. *Beyond prediction: interpretable machine learning and formal testing of renal–vascular contributions to the ankle–brachial index in type 2 diabetes.* Frontiers in Cardiovascular Medicine (under review, manuscript 1996174).

The repository reproduces every number, table and figure in the paper: cohort derivation, missing-value
imputation, repeated nested cross-validation of four learners (XGBoost, random forest, elastic net, MARS),
model evaluation, permutation importance, SHAP (including interaction values and bootstrap stability),
partial dependence, Friedman's H-statistic, and the regression tests of the MCR × renal-function interaction.

## Data

The individual-level data come from the Taiwan MJ Cohort and were provided by the MJ Health Research Foundation
(authorization MJHRF2025011A). They are **not included** here because their use is governed by a data use
agreement. Researchers can apply for access to the Foundation. The code expects the full-precision extract
`2022_abi_stat.xlsx` (687 records, with case numbers). Do not use rounded/de-identified exports of the same data
(e.g., a 680-row file without case numbers in which creatinine is rounded to integers): duplicate records cannot be
identified in such files and creatinine/eGFR-based analyses are not meaningful. The extract is expected with one row per record and the columns, in order:

`case number, sex (1 = male, 2 = female), age, Duration, smoking, drinking, HbA1c, TG, HDL, LDL, ALT, Creatinine,
ma (microalbumin-to-creatinine ratio, mg/g), in (fasting insulin), SBP, DBP, bmi, abi`

## Computational environment

* Python 3.13 — packages pinned in `requirements.txt` (`pip install -r requirements.txt`)
* R 4.3.3 with `caret` 6.0-94 and `earth` 5.3.3 (`Rscript R/install_packages.R`) — used for MARS
* Tested on Linux (Ubuntu 24.04), 2 CPU cores, 8 GB RAM. All random seeds are fixed (`SEED = 20261101` in `scripts/common.py`).

## Reproducing the analysis

```bash
pip install -r requirements.txt
Rscript R/install_packages.R
bash run_all.sh /path/to/2022_abi_stat.xlsx results
```

The main nested cross-validation takes about 1–2 hours on two cores. For a quick functional check of the
whole pipeline (about 20 minutes; numbers will not match the paper exactly), run:

```bash
QUICK=1 bash run_all.sh /path/to/2022_abi_stat.xlsx test_output
```

## Outputs

| Folder | Content |
|---|---|
| `results/cohort_flow.json` | Participant flow counts (Figure 1) |
| `results/main/` | Fold-level metrics, out-of-fold predictions, selected hyperparameters (Table 2, Figure 3) |
| `results/duration/` | Sensitivity analysis with diabetes duration (Table S1) |
| `results/interp/` | SHAP values, bootstrap ranks, Friedman's H, MARS partial dependence (Figures 4–5, S1) |
| `results/reg/` | Multivariable and interaction regression models (Tables 3 and 5) |
| `results/tables/` | Tables 1, 2, 4 and S1 as CSV |
| `results/figs/` | Figures 1–5 and S1 (600-dpi TIFF/PNG and PDF) |

## Pipeline

| Step | Script | What it does | Paper |
|---|---|---|---|
| 1 | `scripts/01_prepare_cohort.py` | Excludes missing/implausible ABI and duplicate records (case number), requires the core panel (age, HbA1c, TG, LDL-C, ALT, creatinine, MCR, SBP, DBP); computes eGFR (CKD-EPI 2021) | Fig 1, Table 1 |
| 2 | `scripts/02_nested_cv.py` | Outer 5-fold CV stratified by ABI quartile × 10 repeats. Within each outer training fold only: random-forest iterative imputation (outcome not used), z-standardization, inner 5-fold grid search. Evaluates XGBoost, RF, elastic net and a null model on the untouched held-out fold; RF permutation importance on the held-out fold; elastic-net coefficients | Table 2, Table 4, Fig 3–4 |
| 3 | `R/03_mars_nested.R` | MARS tuned (inner 5-fold) and evaluated on exactly the same outer folds, imputation and scaling | Table 2, Fig 3 |
| 4 | `scripts/04_interpretation.py` | Full-cohort XGBoost (tuned by repeated CV): TreeSHAP values, SHAP interaction values, 200 bootstrap refits for rank stability, Friedman's H for MCR × creatinine with a 100-permutation null; full-cohort elastic net | Fig 4–5, Fig S1 |
| 5 | `R/06_mars_full.R` | Full-cohort MARS, selected terms and partial dependence | Fig 5D |
| 6 | `scripts/05_regression.py` | Multivariable OLS with multiple imputation (m = 20, Rubin's rules); MCR × log-creatinine and MCR × eGFR interaction models (complete core data with HC3 SEs; extended adjustment with MI; all 673 participants with MI) | Table 3, Table 5, Fig 5C |
| 7 | `scripts/08_tables.py`, `09_sensitivity_table.py` | Tables 1, 2, 4 and Table S1 (diabetes duration added as a predictor) | Tables |
| 8 | `scripts/07a_schematics.py`, `07b_results_figures.py` | Figures 1–5 and S1 (600 dpi TIFF/PNG and PDF) | Figures |

### Hyper-parameter grids (inner 5-fold CV, RMSE criterion)

| Model | Grid | Combinations |
|---|---|---|
| XGBoost (300 rounds) | learning rate {0.01, 0.1, 0.3} × max depth {2, 4, 8} × subsample {0.6, 0.8, 1.0} | 27 |
| Random forest | trees {500, 1000} × mtry {4, 7, 10, 13} | 8 |
| Elastic net | α (L1 ratio) {0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0} × 100-value λ path | 700 |
| MARS (nk = 20) | degree {1, 2} × nprune {2, 4, 6, 8, 10, 12, 15, 20} | 16 |

### Leakage safeguards

Imputation, standardization and tuning are fitted inside each outer training fold; the held-out fold is used
once, for evaluation only. Duplicate exports of the same participant are removed before splitting, so no
person contributes to both training and test data.

## Licence

MIT (code only).
