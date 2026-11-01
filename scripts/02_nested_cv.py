"""Repeated nested cross-validation for continuous ABI.

Outer loop : 5-fold CV stratified by ABI quartile, repeated R times (default 10 -> 50 outer folds).
Within each outer training fold (never touching the held-out fold):
  * missForest-style random-forest iterative imputation of predictors (outcome not used),
  * z-score standardization,
  * hyperparameter tuning by inner 5-fold CV (exhaustive grid, RMSE criterion).
The tuned model is refitted on the whole outer training fold and evaluated once on the untouched
held-out fold. MARS is tuned and evaluated on the same folds in R (03_mars_nested.R).

Usage: python 02_nested_cv.py <analytic_cohort.csv> <out_dir> [--repeats R] [--with-duration] [--tag name]
"""
import argparse, json, os, time
import warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV, KFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.exceptions import ConvergenceWarning
from common import PREDICTORS, SEED, OUTER_SPLITS, INNER_SPLITS

# IterativeImputer may stop at max_iter before its tolerance is met; this does not affect validity.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

XGB_GRID = {"learning_rate": [0.01, 0.1, 0.3], "max_depth": [2, 4, 8], "subsample": [0.6, 0.8, 1.0]}
RF_GRID = {"n_estimators": [500, 1000], "max_features": [4, 7, 10, 13]}   # mtry from ~sqrt(p) to p
EN_L1 = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]                              # alpha (mixing) grid; 100-value lambda path


def make_imputer(seed):
    rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1)
    return IterativeImputer(estimator=rf, max_iter=10, random_state=seed, skip_complete=True)


def metrics(y, p):
    sse, sst = np.sum((y - p) ** 2), np.sum((y - y.mean()) ** 2)
    return {"rmse": float(np.sqrt(mean_squared_error(y, p))), "mae": float(mean_absolute_error(y, p)),
            "r2": float(1 - sse / sst)}


def run_fold(k, tr, te, X, y, feats, fold_dir):
    seed = SEED + k
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
    imp = make_imputer(seed).fit(Xtr)
    Xtr_i = pd.DataFrame(imp.transform(Xtr), columns=feats)
    Xte_i = pd.DataFrame(imp.transform(Xte), columns=feats)
    sc = StandardScaler().fit(Xtr_i)
    Ztr = pd.DataFrame(sc.transform(Xtr_i), columns=feats)
    Zte = pd.DataFrame(sc.transform(Xte_i), columns=feats)
    # files for the R/MARS arm (identical folds, imputation and scaling)
    Ztr.assign(abi=ytr).to_csv(f"{fold_dir}/fold{k:02d}_train.csv", index=False)
    Zte.assign(abi=yte).to_csv(f"{fold_dir}/fold{k:02d}_test.csv", index=False)

    inner = KFold(INNER_SPLITS, shuffle=True, random_state=seed)
    res, preds, extra = [], {}, {}
    # null model: training-fold mean
    preds["Null"] = np.full(len(yte), ytr.mean())
    # XGBoost
    g = GridSearchCV(XGBRegressor(n_estimators=300, objective="reg:squarederror", n_jobs=1, random_state=seed,
                                  verbosity=0), XGB_GRID, cv=inner, scoring="neg_root_mean_squared_error")
    g.fit(Ztr, ytr); preds["XGBoost"] = g.predict(Zte); extra["XGBoost_params"] = g.best_params_
    # Random forest (+ held-out permutation importance)
    g = GridSearchCV(RandomForestRegressor(random_state=seed, n_jobs=1), RF_GRID, cv=inner,
                     scoring="neg_root_mean_squared_error")
    g.fit(Ztr, ytr); preds["Random Forest"] = g.predict(Zte); extra["RF_params"] = g.best_params_
    base_mse = mean_squared_error(yte, preds["Random Forest"])
    pi = permutation_importance(g.best_estimator_, Zte, yte, scoring="neg_mean_squared_error",
                                n_repeats=20, random_state=seed, n_jobs=1)
    extra["RF_perm_incMSE_pct"] = dict(zip(feats, (100 * pi.importances_mean / base_mse).tolist()))
    # Elastic net
    en = ElasticNetCV(l1_ratio=EN_L1, alphas=100, cv=inner, random_state=seed, max_iter=50000).fit(Ztr, ytr)
    preds["Elastic Net"] = en.predict(Zte)
    extra["EN_params"] = {"l1_ratio": float(en.l1_ratio_), "lambda": float(en.alpha_)}
    extra["EN_coef"] = dict(zip(feats, en.coef_.tolist()))
    for m, p in preds.items():
        res.append({"fold": k, "model": m, **metrics(yte, p)})
    oof = pd.DataFrame({"fold": k, "row": te, "abi": yte, **{m: p for m, p in preds.items()}})
    return res, oof, {"fold": k, **extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cohort"); ap.add_argument("out")
    ap.add_argument("--repeats", type=int, default=10); ap.add_argument("--with-duration", action="store_true")
    ap.add_argument("--tag", default="main"); ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    d = pd.read_csv(a.cohort)
    feats = PREDICTORS + (["duration"] if a.with_duration else [])
    X, y = d[feats], d["abi"].values
    strata = pd.qcut(y, 4, labels=False)
    out = f"{a.out}/{a.tag}"; fold_dir = f"{out}/folds"; os.makedirs(fold_dir, exist_ok=True)
    cv = RepeatedStratifiedKFold(n_splits=OUTER_SPLITS, n_repeats=a.repeats, random_state=SEED)
    splits = list(cv.split(X, strata))
    pd.DataFrame([{"fold": k, "repeat": k // OUTER_SPLITS, "row": i} for k, (_, te) in enumerate(splits) for i in te]
                 ).to_csv(f"{out}/fold_assignment.csv", index=False)
    t = time.time()
    outs = Parallel(n_jobs=a.jobs, verbose=5)(delayed(run_fold)(k, tr, te, X, y, feats, fold_dir)
                                              for k, (tr, te) in enumerate(splits))
    pd.DataFrame([r for o in outs for r in o[0]]).to_csv(f"{out}/fold_metrics_py.csv", index=False)
    pd.concat([o[1] for o in outs]).to_csv(f"{out}/oof_predictions_py.csv", index=False)
    json.dump([o[2] for o in outs], open(f"{out}/fold_details_py.json", "w"), indent=1)
    json.dump({"grids": {"XGBoost": XGB_GRID, "Random Forest": RF_GRID, "Elastic Net l1_ratio": EN_L1},
               "n_combinations": {"XGBoost": 27, "Random Forest": 8, "Elastic Net": len(EN_L1) * 100},
               "outer": [OUTER_SPLITS, a.repeats], "inner": INNER_SPLITS, "features": feats,
               "seconds": time.time() - t}, open(f"{out}/settings_py.json", "w"), indent=1)


if __name__ == "__main__":
    main()
