"""Build the analytic cohort from the MJ Health Research Foundation extract.

Usage: python 01_prepare_cohort.py <path/to/2022_abi_stat.xlsx> <output_dir>
The extract itself is not distributed (data-use agreement MJHRF2025011A).
"""
import json, sys
import numpy as np, pandas as pd
from common import CORE_PANEL, egfr_ckdepi2021

src, out = sys.argv[1], sys.argv[2]
raw = pd.read_excel(src)
raw.columns = ["case_id", "sex", "age", "duration", "smoking", "drinking", "hba1c", "tg",
               "hdl", "ldl", "alt", "creatinine", "mcr", "insulin", "sbp", "dbp", "bmi", "abi"]
flow = {"records_received": len(raw)}

d = raw[raw["abi"].notna()].copy();                 flow["excluded_no_abi"] = len(raw) - len(d)
n0 = len(d); d = d[d["abi"] <= 2.0];                flow["excluded_implausible_abi"] = n0 - len(d)
n0 = len(d)
dup = d["case_id"].notna() & d["case_id"].duplicated(keep="first")
d = d[~dup];                                         flow["excluded_duplicate_records"] = n0 - len(d)
flow["unique_participants_with_abi"] = len(d)

# implausible laboratory values -> missing
flow["hba1c_set_missing_gt20"] = int((d["hba1c"] > 20).sum())
d.loc[d["hba1c"] > 20, "hba1c"] = np.nan

n0 = len(d); d = d[d[CORE_PANEL].notna().all(axis=1)].copy()
flow["excluded_incomplete_core_panel"] = n0 - len(d)
flow["analytic_cohort"] = len(d)

d["sex"] = (d["sex"] == 2).astype(int)               # 1 = female, 0 = male
d["egfr"] = egfr_ckdepi2021(d["creatinine"].values, d["age"].values, d["sex"].values == 1)
d = d.drop(columns=["smoking", "drinking"]).reset_index(drop=True)
d.to_csv(f"{out}/analytic_cohort.csv", index=False)
json.dump(flow, open(f"{out}/cohort_flow.json", "w"), indent=2)
print(json.dumps(flow, indent=2))
print("missing after cohort definition:", d.isna().sum()[d.isna().sum() > 0].to_dict())
