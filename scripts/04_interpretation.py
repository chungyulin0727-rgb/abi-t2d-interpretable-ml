"""Model interpretation on the full analytic cohort, with stability and interaction analyses.

* Predictors imputed once (missForest-style RF iterative imputation, outcome not used) and standardised.
* XGBoost tuned by repeated 5-fold CV on the full cohort, then interpreted with TreeSHAP
  (global mean |SHAP|, dependence values, SHAP interaction values).
* Stability: 200 bootstrap refits -> distribution of SHAP-importance ranks.
* Friedman's H-statistic for MCR x creatinine, with a permutation null (outcome permuted, model refitted).
* Elastic net refitted on the full cohort (coefficients).
Usage: python 04_interpretation.py <analytic_cohort.csv> <out_dir> [--boot 200] [--hperm 100]
"""
import argparse, json
import warnings
import numpy as np, pandas as pd, shap
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.exceptions import ConvergenceWarning
from common import PREDICTORS, SEED

# IterativeImputer may stop at max_iter before its tolerance is met; this does not affect validity.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

XGB_GRID = {"learning_rate": [0.01, 0.1, 0.3], "max_depth": [2, 4, 8], "subsample": [0.6, 0.8, 1.0]}


def xgb(params, seed):
    return XGBRegressor(n_estimators=300, objective="reg:squarederror", n_jobs=2, random_state=seed,
                        verbosity=0, **params)


def friedman_h(model, X, j, k, n_points=150, seed=0):
    """Two-way H statistic (Friedman & Popescu 2008) evaluated at a sample of observed points."""
    rng = np.random.default_rng(seed)
    pts = X.iloc[rng.choice(len(X), size=min(n_points, len(X)), replace=False)]
    Xa = X.values

    def pd_at(cols, vals):
        Z = np.tile(Xa, (len(vals), 1))
        for c_i, c in enumerate(cols):
            Z[:, X.columns.get_loc(c)] = np.repeat(vals[:, c_i], len(Xa))
        return model.predict(pd.DataFrame(Z, columns=X.columns)).reshape(len(vals), len(Xa)).mean(axis=1)

    pjk = pd_at([j, k], pts[[j, k]].values); pj = pd_at([j], pts[[j]].values); pk = pd_at([k], pts[[k]].values)
    pjk, pj, pk = pjk - pjk.mean(), pj - pj.mean(), pk - pk.mean()
    return float(np.sqrt(np.sum((pjk - pj - pk) ** 2) / np.sum(pjk ** 2)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cohort"); ap.add_argument("out")
    ap.add_argument("--boot", type=int, default=200); ap.add_argument("--hperm", type=int, default=100)
    a = ap.parse_args()
    d = pd.read_csv(a.cohort); y = d["abi"].values
    imp = IterativeImputer(estimator=RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=2),
                           max_iter=10, random_state=SEED, skip_complete=True)
    Xi = pd.DataFrame(imp.fit_transform(d[PREDICTORS]), columns=PREDICTORS)
    X = pd.DataFrame(StandardScaler().fit_transform(Xi), columns=PREDICTORS)
    Xi.assign(abi=y).to_csv(f"{a.out}/full_imputed_unscaled.csv", index=False)
    X.assign(abi=y).to_csv(f"{a.out}/full_imputed_scaled.csv", index=False)

    g = GridSearchCV(xgb({}, SEED), XGB_GRID, cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED),
                     scoring="neg_root_mean_squared_error").fit(X, y)
    bp = g.best_params_; model = xgb(bp, SEED).fit(X, y)
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X)
    imp_shap = pd.Series(np.abs(sv).mean(axis=0), index=PREDICTORS).sort_values(ascending=False)
    pd.DataFrame(sv, columns=PREDICTORS).to_csv(f"{a.out}/shap_values.csv", index=False)
    siv = expl.shap_interaction_values(X)
    ij = (PREDICTORS.index("mcr"), PREDICTORS.index("creatinine"))
    mcr_cre_int = np.abs(2 * siv[:, ij[0], ij[1]]).mean()
    total_abs = np.abs(siv).sum(axis=(1, 2)).mean()
    # share of MCR's own attribution that is due to its interaction with creatinine
    mcr_total = np.abs(sv[:, ij[0]]).mean()
    pd.DataFrame({"mcr": Xi["mcr"], "creatinine": Xi["creatinine"], "shap_mcr": sv[:, ij[0]],
                  "shap_int_mcr_cre": 2 * siv[:, ij[0], ij[1]]}).to_csv(f"{a.out}/shap_mcr_dependence.csv", index=False)

    # bootstrap stability of SHAP importance ranks
    rng = np.random.default_rng(SEED); ranks = []
    for b in range(a.boot):
        idx = rng.integers(0, len(y), len(y))
        m = xgb(bp, SEED + b).fit(X.iloc[idx], y[idx])
        s = np.abs(shap.TreeExplainer(m).shap_values(X)).mean(axis=0)
        ranks.append(pd.Series(s, index=PREDICTORS).rank(ascending=False))
    R = pd.DataFrame(ranks)
    R.to_csv(f"{a.out}/shap_bootstrap_ranks.csv", index=False)

    # Friedman H for MCR x creatinine with permutation null
    H_obs = friedman_h(model, X, "mcr", "creatinine", seed=SEED)
    H_null = []
    for b in range(a.hperm):
        yp = rng.permutation(y)
        H_null.append(friedman_h(xgb(bp, SEED + 1000 + b).fit(X, yp), X, "mcr", "creatinine", seed=SEED))
    H_null = np.array(H_null)

    en = ElasticNetCV(l1_ratio=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], alphas=100, cv=5, random_state=SEED,
                      max_iter=50000).fit(X, y)
    pd.Series(en.coef_, index=PREDICTORS).to_csv(f"{a.out}/en_full_coef.csv", header=["coef"])

    out = {"xgb_best_params": bp, "xgb_cv_rmse_full_tuning": float(-g.best_score_),
           "shap_mean_abs": imp_shap.to_dict(),
           "shap_rank_median": R.median().to_dict(),
           "shap_rank_p2.5": R.quantile(0.025).to_dict(), "shap_rank_p97.5": R.quantile(0.975).to_dict(),
           "shap_top5_freq": (R <= 5).mean().to_dict(),
           "mcr_cre_shap_interaction_mean_abs": float(mcr_cre_int),
           "mcr_mean_abs_shap": float(mcr_total), "mean_total_abs_shap_interaction_matrix": float(total_abs),
           "friedman_H_mcr_cre": H_obs, "friedman_H_null_mean": float(H_null.mean()),
           "friedman_H_null_p95": float(np.quantile(H_null, 0.95)),
           "friedman_H_perm_p": float((1 + np.sum(H_null >= H_obs)) / (1 + len(H_null))),
           "en_l1_ratio": float(en.l1_ratio_), "en_lambda": float(en.alpha_),
           "en_coef": dict(zip(PREDICTORS, en.coef_.tolist())), "n_boot": a.boot, "n_hperm": a.hperm}
    json.dump(out, open(f"{a.out}/interpretation.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ["xgb_best_params", "friedman_H_mcr_cre", "friedman_H_null_p95",
                                          "friedman_H_perm_p"]}, indent=1))
    print(imp_shap.round(4).to_string())


if __name__ == "__main__":
    main()
