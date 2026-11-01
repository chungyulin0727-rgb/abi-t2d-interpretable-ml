"""Tables 1, 2 and 4 and a summary of key numbers. Usage: python 08_tables.py <cohort.csv> <run_dir> <out_dir>"""
import json, sys
import numpy as np, pandas as pd
from common import LABELS

coh, run, out = sys.argv[1:4]
d = pd.read_csv(coh)
# ---------------- Table 1 ----------------
spec = [("age", "Age", "years", "mean"), ("female", "Sex, female", "n (%)", "n"), ("duration", "Diabetes duration", "years", "mean"),
        ("bmi", "Body mass index", "kg/m²", "mean"), ("hba1c", "Glycated hemoglobin", "%", "mean"),
        ("tg", "Triglycerides", "mg/dL", "median"), ("hdl", "High-density lipoprotein cholesterol", "mg/dL", "mean"),
        ("ldl", "Low-density lipoprotein cholesterol", "mg/dL", "mean"), ("alt", "Alanine aminotransferase", "U/L", "median"),
        ("creatinine", "Creatinine", "mg/dL", "median"), ("egfr", "eGFR (CKD-EPI 2021)", "mL/min/1.73 m²", "mean"),
        ("mcr", "Microalbumin-to-creatinine ratio", "mg/g", "median"), ("insulin", "Insulin", "μU/mL", "median"),
        ("sbp", "Systolic blood pressure", "mmHg", "mean"), ("dbp", "Diastolic blood pressure", "mmHg", "mean"),
        ("abi", "Ankle–brachial index", "ratio", "mean")]
d["female"] = d["sex"]
rows = []
for v, lab, unit, kind in spec:
    x = d[v].dropna(); miss = int(d[v].isna().sum())
    if kind == "mean": val = f"{x.mean():.2f} ± {x.std():.2f}"
    elif kind == "median": val = f"{x.median():.2f} [{x.quantile(.25):.2f}–{x.quantile(.75):.2f}]"
    else: val = f"{int(x.sum())} ({100*x.mean():.1f}%)"
    rows.append({"Variable": lab, "Unit": unit, "Value": val, "Missing, n (%)": f"{miss} ({100*miss/len(d):.1f}%)"})
t1 = pd.DataFrame(rows); t1.to_csv(f"{out}/table1.csv", index=False)
abi_cat = {"<0.90": int((d.abi < 0.9).sum()), "0.90-1.40": int(((d.abi >= 0.9) & (d.abi <= 1.4)).sum()), ">1.40": int((d.abi > 1.4).sum())}

# ---------------- Table 2 ----------------
fm = pd.concat([pd.read_csv(f"{run}/fold_metrics_py.csv"), pd.read_csv(f"{run}/fold_metrics_mars.csv")])
null = fm[fm.model == "Null"].set_index("fold")
rows = []
for m in ["XGBoost", "Random Forest", "Elastic Net", "MARS", "Null"]:
    s = fm[fm.model == m].set_index("fold").sort_index()
    r = {"Model": m}
    for k in ["rmse", "mae", "r2"]:
        r[k] = s[k].mean(); r[k + "_sd"] = s[k].std(); r[k + "_lo"] = s[k].quantile(0.025); r[k + "_hi"] = s[k].quantile(0.975)
    diff = s["rmse"] - null.loc[s.index, "rmse"]
    r["drmse"] = diff.mean(); r["drmse_lo"] = diff.quantile(0.025); r["drmse_hi"] = diff.quantile(0.975)
    r["pct_folds_better_than_null"] = float((diff < 0).mean() * 100)
    rows.append(r)
t2 = pd.DataFrame(rows); t2.to_csv(f"{out}/table2_raw.csv", index=False)
# pooled out-of-fold R2 per repeat (each participant predicted once per repeat)
oof = pd.read_csv(f"{run}/oof_predictions_py.csv").sort_values(["fold", "row"]).reset_index(drop=True)
mo = pd.read_csv(f"{run}/oof_predictions_mars.csv").sort_values(["fold", "idx"]).reset_index(drop=True)
oof["MARS"] = mo["MARS"].values; oof["repeat"] = oof["fold"] // 5
pooled = {}
for m in ["XGBoost", "Random Forest", "Elastic Net", "MARS", "Null"]:
    r2 = oof.groupby("repeat").apply(lambda g: 1 - np.sum((g.abi - g[m]) ** 2) / np.sum((g.abi - g.abi.mean()) ** 2))
    pooled[m] = [float(r2.mean()), float(r2.min()), float(r2.max())]

# ---------------- Table 4 ----------------
det = json.load(open(f"{run}/fold_details_py.json"))
pi = pd.DataFrame([x["RF_perm_incMSE_pct"] for x in det])
rk = pi.rank(axis=1, ascending=False)
pos = pi.clip(lower=0)
t4 = pd.DataFrame({"median": pi.median(), "p10": pi.quantile(0.1), "p90": pi.quantile(0.9),
                   "share": pos.mean() / pos.mean().sum(), "pct_positive": (pi > 0).mean() * 100,
                   "median_rank": rk.median(), "top3_pct": (rk <= 3).mean() * 100}).sort_values("median", ascending=False)
t4.index = [LABELS[i] for i in t4.index]; t4.to_csv(f"{out}/table4_raw.csv")
# rank agreement between folds (mean pairwise Spearman of importance vectors)
cm = pi.T.corr(method="spearman").values; iu = np.triu_indices_from(cm, 1)
params = {"XGBoost": pd.DataFrame([x["XGBoost_params"] for x in det]).astype(str).agg(", ".join, axis=1).value_counts().head(3).to_dict(),
          "RF": pd.DataFrame([x["RF_params"] for x in det]).astype(str).agg(", ".join, axis=1).value_counts().head(3).to_dict(),
          "EN_l1": pd.Series([x["EN_params"]["l1_ratio"] for x in det]).value_counts().to_dict(),
          "EN_lambda_median": float(np.median([x["EN_params"]["lambda"] for x in det]))}
enc = pd.DataFrame([x["EN_coef"] for x in det]); sel = (enc.abs() > 1e-10).mean() * 100
mars_bp = pd.read_csv(f"{run}/mars_best_params.csv")
summary = {"n": len(d), "female": int(d.sex.sum()), "abi_categories": abi_cat, "pooled_oof_r2": pooled,
           "rf_perm_mean_pairwise_spearman": float(np.nanmean(cm[iu])), "best_params": params,
           "en_selection_pct": sel.round(1).to_dict(),
           "mars_best": mars_bp[["degree", "nprune"]].astype(str).agg(", ".join, axis=1).value_counts().head(4).to_dict()}
json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
pd.set_option("display.width", 220)
print(t1.to_string(index=False)); print(t2.round(4).to_string(index=False)); print(t4.round(3).to_string())
print(json.dumps(summary, indent=1))
