"""Supplementary Table S1: performance with vs without diabetes duration on the same 10 outer folds.
Usage: python 09_sensitivity_table.py <main_run_dir> <duration_run_dir> <out_csv>"""
import sys
import pandas as pd
main, dur, out = sys.argv[1:4]
rd = lambda r: pd.concat([pd.read_csv(f"{r}/fold_metrics_py.csv"), pd.read_csv(f"{r}/fold_metrics_mars.csv")])
a, b = rd(main), rd(dur)
a = a[a.fold < b.fold.max() + 1]          # identical seeds -> identical first 10 outer folds
rows = []
for m in ["XGBoost", "Random Forest", "Elastic Net", "MARS", "Null"]:
    for lab, t in [("13 predictors", a), ("13 predictors + diabetes duration", b)]:
        s = t[t.model == m]
        rows.append({"Model": m, "Predictor set": lab, "RMSE": f"{s.rmse.mean():.4f} ± {s.rmse.std():.4f}",
                     "MAE": f"{s.mae.mean():.4f} ± {s.mae.std():.4f}", "R2": f"{s.r2.mean():.3f} ± {s.r2.std():.3f}"})
pd.DataFrame(rows).to_csv(out, index=False); print(pd.DataFrame(rows).to_string(index=False))
