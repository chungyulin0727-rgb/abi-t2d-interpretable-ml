"""Shared settings for the ABI–T2D machine-learning analysis."""
import numpy as np

SEED = 20261101
PREDICTORS = ["age", "sex", "bmi", "sbp", "dbp", "tg", "hdl", "ldl", "alt",
              "insulin", "hba1c", "creatinine", "mcr"]
CORE_PANEL = ["age", "hba1c", "tg", "ldl", "alt", "creatinine", "mcr", "sbp", "dbp"]
LABELS = {"age": "Age", "sex": "Sex", "bmi": "BMI", "sbp": "SBP", "dbp": "DBP",
          "tg": "TG", "hdl": "HDL-C", "ldl": "LDL-C", "alt": "ALT", "insulin": "Insulin",
          "hba1c": "HbA1c", "creatinine": "Creatinine", "mcr": "MCR", "duration": "Duration"}
OUTER_SPLITS, OUTER_REPEATS, INNER_SPLITS = 5, 10, 5


def egfr_ckdepi2021(scr, age, female):
    """Race-free CKD-EPI 2021 creatinine equation (mL/min/1.73 m2)."""
    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.241, -0.302)
    return (142 * np.minimum(scr / k, 1) ** a * np.maximum(scr / k, 1) ** -1.200
            * 0.9938 ** age * np.where(female, 1.012, 1.0))
