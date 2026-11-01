"""Figure 1 (participant flow) and Figure 2 (analysis workflow). Usage: python 07a_schematics.py <out_dir> <cohort_flow.json>"""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

out, flow = sys.argv[1], json.load(open(sys.argv[2]))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
INK, MUTED, LINE = "#0b0b0b", "#52514e", "#3a3a38"
BLUE, BLUE_BG, GREY_BG = "#2a78d6", "#e8f0fb", "#f3f3f1"
MODEL = {"XGBoost": "#2a78d6", "Random Forest": "#eb6834", "Elastic Net": "#1baf7a", "MARS": "#eda100"}


def save(fig, name):
    fig.savefig(f"{out}/{name}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(f"{out}/{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{out}/{name}.pdf", bbox_inches="tight")


def box(ax, x, y, w, h, text, fc="white", ec=LINE, lw=1.2, bold=False, size=9, color=INK, align="center"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=fc, ec=ec, lw=lw))
    ha = "center" if align == "center" else "left"
    tx = x if align == "center" else x - w / 2 + 0.018
    ax.text(tx, y, text, ha=ha, va="center", fontsize=size, color=color, fontweight="bold" if bold else "normal",
            linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color=LINE):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, lw=1.2, color=color,
                                 shrinkA=0, shrinkB=0))


# ---------------- Figure 1: participant flow ----------------
fig, ax = plt.subplots(figsize=(7.09, 6.6)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
MX, MW = 0.285, 0.55          # main column centre and width
SX, SW = 0.795, 0.39          # side (exclusion) column centre and width
BH = 0.12                    # main box height
ys = [0.91, 0.655, 0.40, 0.165]
main_txt = [f"Adults with type 2 diabetes\nand an ABI examination\nn = {flow['records_received']} records",
            f"Unique participants with a valid ABI\nn = {flow['unique_participants_with_abi']}",
            f"Analytic cohort\nn = {flow['analytic_cohort']}",
            "Missing BMI, HDL-C and insulin imputed\nwithin each training fold\n(random-forest iterative imputation)"]
styles = [dict(fc=GREY_BG), dict(fc=GREY_BG), dict(fc=BLUE_BG, ec=BLUE, lw=1.6, bold=True), dict(fc="white")]
for y, t, st in zip(ys, main_txt, styles):
    box(ax, MX, y, MW, BH, t, size=8.5, **st)
for y0, y1 in zip(ys[:-1], ys[1:]):
    arrow(ax, MX, y0 - BH / 2 - 0.006, MX, y1 + BH / 2 + 0.006)
ex = [(0.7825, 0.15, f"Excluded (n = {flow['records_received'] - flow['unique_participants_with_abi']})\n"
                    f"•  ABI missing (n = {flow['excluded_no_abi']})\n•  Implausible ABI > 2.0 (n = {flow['excluded_implausible_abi']})\n"
                    f"•  Duplicate records (n = {flow['excluded_duplicate_records']})"),
      (0.5275, 0.16, f"Excluded (n = {flow['excluded_incomplete_core_panel']})\nIncomplete core panel*:\nage, HbA1c, TG, LDL-C, ALT,\ncreatinine, MCR, SBP or DBP")]
for y, h, t in ex:
    box(ax, SX, y, SW, h, t, align="left", size=8)
    ax.plot([MX, SX - SW / 2 - 0.006], [y, y], color=LINE, lw=1.2)
ax.text(0.01, 0.0, "*HbA1c > 20%% was treated as missing (n = %d).\nABI, ankle–brachial index; MCR, microalbumin-to-creatinine ratio."
        % flow["hba1c_set_missing_gt20"], fontsize=7.5, color=MUTED, va="bottom")
save(fig, "Figure1_flowchart")
plt.close(fig)

# ---------------- Figure 2: workflow ----------------
fig, ax = plt.subplots(figsize=(7.09, 8.2)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
stages = [(0.95, "1  Data preparation", f"n = {flow['analytic_cohort']}  ·  13 prespecified predictors (no feature selection)  ·  outcome: ABI"),
          (0.845, "2  Outer resampling", "5-fold CV stratified by ABI quartile, repeated 10 times (50 held-out folds)")]
for y, t, s in stages:
    box(ax, 0.5, y, 0.94, 0.075, "", fc=GREY_BG, ec=LINE)
    ax.text(0.05, y + 0.014, t, fontsize=9.5, fontweight="bold", color=INK, va="center")
    ax.text(0.05, y - 0.016, s, fontsize=8, color=MUTED, va="center")
arrow(ax, 0.5, 0.9105, 0.5, 0.8835)
# within-training-fold container
ax.add_patch(FancyBboxPatch((0.03, 0.315), 0.94, 0.45, boxstyle="round,pad=0.008,rounding_size=0.015", fc="white",
                            ec=BLUE, lw=1.4, ls=(0, (4, 2))))
ax.text(0.05, 0.745, "3  Within each outer training fold only (held-out fold untouched)", fontsize=9.5, fontweight="bold",
        color=BLUE, va="center")
arrow(ax, 0.5, 0.8075, 0.5, 0.775)
box(ax, 0.5, 0.7, 0.86, 0.045, "Imputation (random forest, outcome not used)  →  z-standardization", fc=BLUE_BG, ec=BLUE, size=8.5)
arrow(ax, 0.5, 0.6775, 0.5, 0.655)
xs = [0.155, 0.385, 0.615, 0.845]
tune = {"XGBoost": "27 combinations\nlearning rate · depth\n· subsample", "Random Forest": "8 combinations\ntrees · mtry",
        "Elastic Net": "7 α × 100 λ\n(regularization path)", "MARS": "16 combinations\ndegree · nprune"}
ax.plot([xs[0], xs[-1]], [0.655, 0.655], color=LINE, lw=1.2)
for x, (m, c) in zip(xs, MODEL.items()):
    arrow(ax, x, 0.655, x, 0.625)
    ax.add_patch(FancyBboxPatch((x - 0.105, 0.43), 0.21, 0.19, boxstyle="round,pad=0.006,rounding_size=0.012", fc="white",
                                ec=c, lw=1.8))
    ax.add_patch(FancyBboxPatch((x - 0.105, 0.585), 0.21, 0.035, boxstyle="round,pad=0.006,rounding_size=0.012", fc=c,
                                ec=c, lw=1.8))
    ax.text(x, 0.6025, m, ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    ax.text(x, 0.505, "Inner 5-fold CV\n" + tune[m], ha="center", va="center", fontsize=7.2, color=INK, linespacing=1.35)
    arrow(ax, x, 0.43, x, 0.395)
ax.plot([xs[0], xs[-1]], [0.395, 0.395], color=LINE, lw=1.2)
box(ax, 0.5, 0.36, 0.86, 0.045, "Refit tuned model on outer training fold  →  predict held-out fold once", fc=BLUE_BG, ec=BLUE, size=8.5)
arrow(ax, 0.5, 0.3075, 0.5, 0.2925)
box(ax, 0.5, 0.25, 0.94, 0.075, "", fc=GREY_BG)
ax.text(0.05, 0.264, "4  Performance", fontsize=9.5, fontweight="bold", va="center")
ax.text(0.05, 0.234, "RMSE, MAE, R² per held-out fold (mean ± SD, 95% interval) · null-model comparison",
        fontsize=7.8, color=MUTED, va="center")
arrow(ax, 0.5, 0.2125, 0.5, 0.1825)
box(ax, 0.5, 0.11, 0.94, 0.14, "", fc=GREY_BG)
ax.text(0.05, 0.155, "5  Interpretation and robustness", fontsize=9.5, fontweight="bold", va="center")
ax.text(0.05, 0.098, "RF permutation importance on held-out folds · XGBoost SHAP with 200 bootstrap refits\n"
        "Elastic-net selection frequency · MARS partial dependence · calibration\n"
        "Formal tests: MCR × creatinine and MCR × eGFR regression · Friedman's H",
        fontsize=7.8, color=MUTED, va="center", linespacing=1.45)
save(fig, "Figure2_workflow")
plt.close(fig)
print("schematics written")
