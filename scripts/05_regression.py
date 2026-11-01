"""Linear regression analyses: standardised coefficients and formal interaction tests.

Table 3 : multivariable OLS of ABI on all 13 predictors plus diabetes duration (standardised),
          multiple imputation (m = 20, chained equations with posterior draws, outcome included),
          pooled by Rubin's rules.
Table 5 : formal tests of the MCR x renal-function interaction requested by the reviewers:
          ABI ~ log(MCR) x log(creatinine)  and  ABI ~ log(MCR) x eGFR  (CKD-EPI 2021),
          (A) core-panel adjustment, complete data (no imputation needed), HC3 robust SEs;
          (B) extended adjustment (+ duration, BMI, HDL-C, insulin) with MI;
          (C) sensitivity: all 673 unique participants with ABI, MI of every missing value.
Usage: python 05_regression.py <analytic_cohort.csv> <stat.xlsx> <out_dir>
"""
import sys
import warnings
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.exceptions import ConvergenceWarning
from common import PREDICTORS, SEED, egfr_ckdepi2021

# IterativeImputer may stop at max_iter before its tolerance is met; this does not affect validity.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

M = 20


def mi_datasets(df, cols, m=M):
    out = []
    for i in range(m):
        imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True, max_iter=20, random_state=SEED + i)
        z = df.copy(); z[cols] = imp.fit_transform(df[cols]); out.append(z)
    return out


def pool(fits, terms):
    Q = np.array([[f.params[t] for t in terms] for f in fits]); U = np.array([[f.bse[t] ** 2 for t in terms] for f in fits])
    qbar, ubar, b = Q.mean(0), U.mean(0), Q.var(0, ddof=1); m = len(fits)
    T = ubar + (1 + 1 / m) * b
    r = (1 + 1 / m) * b / ubar
    dfree = (m - 1) * (1 + 1 / r) ** 2
    se = np.sqrt(T); tval = qbar / se; p = 2 * stats.t.sf(np.abs(tval), dfree)
    ci = stats.t.ppf(0.975, dfree) * se
    return pd.DataFrame({"term": terms, "beta": qbar, "lo": qbar - ci, "hi": qbar + ci, "p": p})


def fit_table3(d):
    cols = PREDICTORS + ["duration", "abi"]
    fits = []
    for z in mi_datasets(d[cols], cols):
        z = (z - z.mean()) / z.std()
        fits.append(smf.ols("abi ~ " + " + ".join(PREDICTORS + ["duration"]), data=z).fit())
    t = pool(fits, PREDICTORS + ["duration"])
    t["r2_mean"] = np.mean([f.rsquared for f in fits])
    return t.sort_values("beta", key=np.abs, ascending=False)


CORE_ADJ = "age + sex + sbp + dbp + tg + ldl + alt + hba1c"
EXT_ADJ = CORE_ADJ + " + duration + bmi + hdl + insulin"


def prep(d):
    d = d.copy(); d["lmcr"] = np.log(d["mcr"]); d["lcre"] = np.log(d["creatinine"])
    d["egfr10"] = d["egfr"] / 10
    for c in ["lmcr", "lcre", "egfr10"]:
        d[c + "_c"] = d[c] - d[c].mean()
    return d


def interaction_rows(d, label, ext=False, mi=False):
    rows = []
    for name, renal in [("log creatinine", "lcre_c"), ("eGFR (per 10 mL/min/1.73 m2)", "egfr10_c")]:
        f = f"abi ~ lmcr_c * {renal} + " + (EXT_ADJ if ext else CORE_ADJ)
        term = f"lmcr_c:{renal}"
        if not mi:
            fit = smf.ols(f, data=d).fit(cov_type="HC3")
            ci = fit.conf_int().loc[term]
            rows.append({"analysis": label, "renal_marker": name, "n": int(fit.nobs), "beta": fit.params[term],
                         "lo": ci[0], "hi": ci[1], "p": fit.pvalues[term],
                         "main_lmcr": fit.params["lmcr_c"], "main_lmcr_p": fit.pvalues["lmcr_c"]})
        else:
            cols = ["abi", "age", "sex", "sbp", "dbp", "tg", "ldl", "alt", "hba1c", "duration", "bmi", "hdl",
                    "insulin", "mcr", "creatinine"]
            fits = []
            for z in mi_datasets(d[cols], cols):
                z["mcr"] = z["mcr"].clip(lower=0.1); z["creatinine"] = z["creatinine"].clip(lower=0.2)
                z["egfr"] = egfr_ckdepi2021(z["creatinine"].values, z["age"].values, z["sex"].round().values == 1)
                z = prep(z)
                fits.append(smf.ols(f, data=z).fit())
            t = pool(fits, [term, "lmcr_c"])
            rows.append({"analysis": label, "renal_marker": name, "n": len(d), "beta": t.beta[0], "lo": t.lo[0],
                         "hi": t.hi[0], "p": t.p[0], "main_lmcr": t.beta[1], "main_lmcr_p": t.p[1]})
    return rows


def main():
    cohort, src, out = sys.argv[1:4]
    d = pd.read_csv(cohort)
    t3 = fit_table3(d); t3.to_csv(f"{out}/table3_ols_mi.csv", index=False)
    d = prep(d)
    rows = interaction_rows(d, "A: core adjustment, n = 565 (complete data)")
    rows += interaction_rows(d, "B: extended adjustment, multiple imputation", ext=True, mi=True)
    # C: all unique participants with ABI (no core-panel restriction), everything imputed
    raw = pd.read_excel(src)
    raw.columns = ["case_id", "sex", "age", "duration", "smoking", "drinking", "hba1c", "tg", "hdl", "ldl", "alt",
                   "creatinine", "mcr", "insulin", "sbp", "dbp", "bmi", "abi"]
    a = raw[raw.abi.notna() & (raw.abi <= 2)]
    a = a[~(a.case_id.notna() & a.case_id.duplicated())].copy()
    a.loc[a.hba1c > 20, "hba1c"] = np.nan; a["sex"] = (a["sex"] == 2).astype(int)
    rows += interaction_rows(a.reset_index(drop=True), f"C: all unique participants with ABI, n = {len(a)}, MI",
                             ext=True, mi=True)
    t5 = pd.DataFrame(rows); t5.to_csv(f"{out}/table5_interaction.csv", index=False)
    pd.set_option("display.width", 200)
    print(t3.round(4).to_string(index=False)); print(t5.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
