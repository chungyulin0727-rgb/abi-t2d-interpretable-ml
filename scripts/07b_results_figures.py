"""Figures 3-5 and supplementary figures from the nested-CV and interpretation outputs.
Usage: python 07b_results_figures.py <run_dir> <interp_dir> <reg_dir> <fig_dir>
"""
import json, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from common import PREDICTORS, LABELS

run, interp, reg, figd = sys.argv[1:5]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.linewidth": 0.8,
                     "xtick.color": "#3a3a38", "ytick.color": "#3a3a38", "axes.labelcolor": "#0b0b0b"})
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"Null": "#9a9893", "XGBoost": "#2a78d6", "Random Forest": "#eb6834", "Elastic Net": "#1baf7a", "MARS": "#eda100"}
ORDER = ["Null", "Elastic Net", "MARS", "Random Forest", "XGBoost"]


def save(fig, name):
    fig.savefig(f"{figd}/{name}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(f"{figd}/{name}.png", dpi=600, bbox_inches="tight"); fig.savefig(f"{figd}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def panel(ax, letter):
    ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


fm = pd.concat([pd.read_csv(f"{run}/fold_metrics_py.csv"), pd.read_csv(f"{run}/fold_metrics_mars.csv")])
oof = pd.read_csv(f"{run}/oof_predictions_py.csv")
mo = pd.read_csv(f"{run}/oof_predictions_mars.csv")
oof = oof.sort_values(["fold", "row"]).reset_index(drop=True)
mo = mo.sort_values(["fold", "idx"]).reset_index(drop=True)
oof["MARS"] = mo["MARS"].values

# ---------------- Figure 3: performance + calibration ----------------
fig, axs = plt.subplots(1, 3, figsize=(7.09, 2.7), gridspec_kw={"width_ratios": [1, 1, 1.05], "wspace": 0.42})
rng = np.random.default_rng(1)
for ax, met, lab in [(axs[0], "rmse", "RMSE (held-out folds)"), (axs[1], "r2", "R² (held-out folds)")]:
    for i, m in enumerate(ORDER):
        v = fm.loc[fm.model == m, met].values
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=7, color=COL[m], alpha=0.45, lw=0)
        q = np.percentile(v, [2.5, 50, 97.5]); mu = v.mean()
        ax.plot([i, i], [q[0], q[2]], color=INK, lw=1.1); ax.plot([i - 0.22, i + 0.22], [mu, mu], color=INK, lw=2)
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(["Null", "EN", "MARS", "RF", "XGB"])
    ax.set_ylabel(lab); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
axs[1].axhline(0, color=MUTED, lw=0.8, ls="--")
panel(axs[0], "A"); panel(axs[1], "B")
# calibration: out-of-fold predictions of the best-performing model (XGBoost), averaged over the 10 repeats
CAL = "XGBoost"
avg = oof.groupby("row")[["abi", "XGBoost", "MARS", "Elastic Net", "Random Forest"]].mean()
ax = axs[2]
ax.scatter(avg[CAL], avg["abi"], s=6, color=COL[CAL], alpha=0.5, lw=0)
bins = pd.qcut(avg[CAL], 10, duplicates="drop")
g = avg.groupby(bins, observed=True).agg(p=(CAL, "mean"), o=("abi", "mean"), se=("abi", lambda s: s.std() / np.sqrt(len(s))))
ax.errorbar(g.p, g.o, yerr=1.96 * g.se, fmt="o", ms=3.5, color=INK, lw=1, capsize=0)
lo, hi = np.floor(avg[CAL].min() * 50) / 50 - 0.01, np.ceil(avg[CAL].max() * 50) / 50 + 0.01
ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.9, ls="--")
cals = {m: smf.ols("abi ~ Q('%s')" % m, data=avg).fit() for m in ["XGBoost", "Random Forest", "Elastic Net", "MARS"]}
cal = cals[CAL]
ax.set_xlim(lo, hi); ax.set_ylim(0.4, 1.8)
ax.set_xlabel("Predicted ABI (XGBoost, out-of-fold)"); ax.set_ylabel("Observed ABI")
ax.text(0.03, 0.97, f"Calibration slope {cal.params.iloc[1]:.2f}\nPrediction range {avg[CAL].min():.2f}–{avg[CAL].max():.2f}",
        transform=ax.transAxes, va="top", fontsize=7.5, color=MUTED)
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); panel(ax, "C")
save(fig, "Figure3_performance_calibration")
json.dump({m: {"slope": float(f.params.iloc[1]), "intercept": float(f.params.iloc[0]), "slope_ci": f.conf_int().iloc[1].tolist(),
               "pred_range": [float(avg[m].min()), float(avg[m].max())]} for m, f in cals.items()},
          open(f"{run}/calibration.json", "w"), indent=1)

# ---------------- Figure 4: importance across approaches ----------------
det = json.load(open(f"{run}/fold_details_py.json"))
pi = pd.DataFrame([d["RF_perm_incMSE_pct"] for d in det])
enc = pd.DataFrame([d["EN_coef"] for d in det])
it = json.load(open(f"{interp}/interpretation.json"))
shapv = pd.read_csv(f"{interp}/shap_values.csv").abs().mean()
R = pd.read_csv(f"{interp}/shap_bootstrap_ranks.csv")
en_full = pd.read_csv(f"{interp}/en_full_coef.csv", index_col=0)["coef"]
fig, axs = plt.subplots(1, 3, figsize=(7.09, 3.6), gridspec_kw={"wspace": 0.75})
# A RF permutation importance
o = pi.median().sort_values().index
ax = axs[0]
for i, f in enumerate(o):
    q = np.percentile(pi[f], [10, 25, 50, 75, 90])
    ax.plot([q[0], q[4]], [i, i], color=COL["Random Forest"], lw=1); ax.plot([q[1], q[3]], [i, i], color=COL["Random Forest"], lw=3.2, solid_capstyle="round")
    ax.plot(q[2], i, "o", color="white", mec=INK, ms=3.5, mew=0.9)
ax.axvline(0, color=MUTED, lw=0.8, ls="--")
ax.set_yticks(range(len(o))); ax.set_yticklabels([LABELS[f] for f in o])
ax.set_xlabel("Increase in held-out MSE (%)\nafter permutation"); ax.grid(axis="x", color=GRID, lw=0.6); panel(ax, "A")
ax.set_title("Random forest", fontsize=9, color=INK)
# B SHAP
o = shapv.sort_values().index
ax = axs[1]
ax.barh(range(len(o)), shapv[o].values * 1000, color=COL["XGBoost"], height=0.62)
for i, f in enumerate(o):
    lo_, hi_ = R[f].quantile(0.025), R[f].quantile(0.975)
    ax.text(shapv[f] * 1000 + shapv.max() * 1000 * 0.04, i, f"{int(lo_)}–{int(hi_)}", va="center", fontsize=6.5, color=MUTED)
ax.set_yticks(range(len(o))); ax.set_yticklabels([LABELS[f] for f in o])
ax.set_xlim(0, shapv.max() * 1000 * 1.45)
ax.set_xlabel("Mean |SHAP| (×10⁻³ ABI units)\nlabel: bootstrap 95% rank range"); ax.grid(axis="x", color=GRID, lw=0.6); panel(ax, "B")
ax.set_title("XGBoost", fontsize=9, color=INK)
# C Elastic net
sel = (enc.abs() > 1e-10).mean()
o = en_full.abs().sort_values().index
ax = axs[2]
ax.barh(range(len(o)), en_full[o].values * 1000, color=COL["Elastic Net"], height=0.62)
for i, f in enumerate(o):
    ax.text(1.02, i, f"{sel[f]*100:.0f}%", transform=ax.get_yaxis_transform(), va="center", fontsize=6.5, color=MUTED)
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_yticks(range(len(o))); ax.set_yticklabels([LABELS[f] for f in o])
ax.set_xlabel("Coefficient (×10⁻³ ABI per SD)\nlabel: selection frequency, 50 folds"); ax.grid(axis="x", color=GRID, lw=0.6); panel(ax, "C")
ax.set_title("Elastic net", fontsize=9, color=INK)
save(fig, "Figure4_feature_importance")

# ---------------- Figure 5: MCR, renal function and nonlinearity ----------------
dep = pd.read_csv(f"{interp}/shap_mcr_dependence.csv")
fig, axs = plt.subplots(2, 2, figsize=(7.09, 5.6), gridspec_kw={"hspace": 0.55, "wspace": 0.38})
ax = axs[0, 0]
sc = ax.scatter(dep["mcr"], dep["shap_mcr"] * 1000, c=dep["creatinine"].clip(0.4, 2.0), cmap="Blues", s=9, lw=0.2,
                edgecolors="#3a3a38", vmin=0.4, vmax=2.0)
ax.set_xscale("log"); ax.axhline(0, color=MUTED, lw=0.8, ls="--")
cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label("Creatinine (mg/dL)", fontsize=7.5); cb.ax.tick_params(labelsize=7)
ax.set_xlabel("MCR (mg/g, log scale)"); ax.set_ylabel("SHAP value for MCR (×10⁻³)"); panel(ax, "A")
ax = axs[0, 1]
ax.scatter(dep["mcr"], dep["shap_int_mcr_cre"] * 1000, s=8, color=COL["XGBoost"], alpha=0.55, lw=0)
ax.set_xscale("log"); ax.axhline(0, color=MUTED, lw=0.8, ls="--")
ax.set_xlabel("MCR (mg/g, log scale)"); ax.set_ylabel("SHAP interaction MCR × creatinine (×10⁻³)"); panel(ax, "B")
# C: regression interaction, predicted ABI across MCR at creatinine 25th/50th/75th centile (model A)
coh = pd.read_csv(f"{run}/../analytic_cohort.csv")
coh["lmcr"] = np.log(coh["mcr"]); coh["lcre"] = np.log(coh["creatinine"])
fit = smf.ols("abi ~ lmcr * lcre + age + sex + sbp + dbp + tg + ldl + alt + hba1c", data=coh).fit(cov_type="HC3")
ax = axs[1, 0]
grid = np.linspace(np.log(coh.mcr.quantile(0.02)), np.log(coh.mcr.quantile(0.98)), 60)
base = coh[["age", "sex", "sbp", "dbp", "tg", "ldl", "alt", "hba1c"]].mean()
for q, c, ls in [(0.25, "#9bc2f0", "-"), (0.5, "#2a78d6", "-"), (0.75, "#103f78", "-")]:
    cre = coh.creatinine.quantile(q)
    nd = pd.DataFrame({"lmcr": grid, "lcre": np.log(cre), **{k: v for k, v in base.items()}})
    pr = fit.get_prediction(nd).summary_frame()
    ax.plot(np.exp(grid), pr["mean"], color=c, lw=1.8, ls=ls, label=f"{cre:.2f} mg/dL (P{int(q*100)})")
    ax.fill_between(np.exp(grid), pr["mean_ci_lower"], pr["mean_ci_upper"], color=c, alpha=0.12, lw=0)
t5 = pd.read_csv(f"{reg}/table5_interaction.csv")
pint = t5.loc[0, "p"]
ax.set_xscale("log"); ax.set_xlabel("MCR (mg/g, log scale)"); ax.set_ylabel("Adjusted predicted ABI")
ax.legend(title="Creatinine", fontsize=7, title_fontsize=7.5, frameon=False, loc="upper left")
ax.text(0.98, 0.97, f"MCR × creatinine\np = {pint:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
panel(ax, "C")
# D: Friedman H vs permutation null summary + MARS partial dependence for DBP
ax = axs[1, 1]
pdp = pd.read_csv(f"{interp}/mars_pdp.csv")
for f in PREDICTORS:
    sdat = pdp[pdp.feature == f]
    if f == "dbp":
        continue
    ax.plot(sdat.x, sdat.yhat, color="#c9c8c3", lw=1.0)
sdat = pdp[pdp.feature == "dbp"]
ax.plot(sdat.x, sdat.yhat, color="#2a78d6", lw=2.0)
ax.text(sdat.x.iloc[-1] - 0.1, sdat.yhat.iloc[-1], "DBP", ha="right", va="center", fontsize=8, color=INK)
ax.text(-2.4, pdp[pdp.feature == "age"].yhat.iloc[0] + 0.002, "12 other predictors\n(not selected; flat)", fontsize=7.5, color=MUTED, va="bottom")
ax.set_xlabel("Predictor value (SD units)"); ax.set_ylabel("MARS partial dependence (ABI)")
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
panel(ax, "D")
save(fig, "Figure5_mcr_renal_nonlinearity")

# ---------------- Supplementary: rank stability ----------------
fig, axs = plt.subplots(1, 2, figsize=(7.09, 3.3), gridspec_kw={"wspace": 0.5})
rk = pi.rank(axis=1, ascending=False)
for ax, M, t, c in [(axs[0], rk, "RF permutation importance\n(50 held-out folds)", COL["Random Forest"]),
                    (axs[1], R, "XGBoost mean |SHAP|\n(200 bootstrap refits)", COL["XGBoost"])]:
    o = M.median().sort_values(ascending=False).index
    ax.boxplot([M[f] for f in o], orientation="horizontal", widths=0.55, patch_artist=True, showfliers=False,
               boxprops=dict(facecolor=c, alpha=0.35, edgecolor=c), medianprops=dict(color=INK), whiskerprops=dict(color=c), capprops=dict(color=c))
    ax.set_yticks(range(1, len(o) + 1)); ax.set_yticklabels([LABELS[f] for f in o]); ax.invert_xaxis()
    ax.set_xlabel("Importance rank (1 = most important)"); ax.set_title(t, fontsize=8.5)
    ax.set_xlim(13.5, 0.5); ax.set_xticks([1, 3, 5, 7, 9, 11, 13]); ax.grid(axis="x", color=GRID, lw=0.6)
panel(axs[0], "A"); panel(axs[1], "B")
save(fig, "FigureS1_rank_stability")
print("result figures written")
