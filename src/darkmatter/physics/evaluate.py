"""WP-E1/E2: evaluation harness + baseline ablations.

All cluster-safe (folds from splits.json; the fitted model never saw
PPMC labels or held-out csat folds).

1. Csat ranking — Spearman rho of predicted log10 csat vs log10 bound,
   per fold and overall (LLPSDB brackets + PhaSePro points).
2. PPMC driver classification — ROC-AUC of -log10 csat_pred
   (lower csat = droplets form at lower concentration = driver).
3. Baselines — single-descriptor scores on the same two tasks.

Run: PYTHONPATH=src python3 -m darkmatter.physics.evaluate
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import spearmanr

from darkmatter.data.ingest import load_ppmc
from darkmatter.physics.calibration import (
    DEFAULT_SALT_MM,
    _features_row,
    assemble_dataset,
    condition_score,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PROCESSED = os.path.join(ROOT, "data", "processed")


def _roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC (ties handled). Larger s => more positive."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney with tie-corrected ranks
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within tie groups
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    r_pos = ranks[y == 1].sum()
    n1, n0 = len(pos), len(neg)
    return float((r_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def run() -> dict:
    cal = json.load(open(os.path.join(PROCESSED, "calibration_result.json")))
    params = np.array(cal["final_fit"]["params"])
    version = cal.get("model_version", "v1")

    data, _ = assemble_dataset()
    pred = np.array([condition_score(d, params) for d in data])
    obs = np.array([np.log10(d["c_uvM"]) for d in data])

    ranking = {}
    for f in sorted({d["out_fold"] for d in data}):
        idx = np.array([k for k, d in enumerate(data) if d["out_fold"] == f])
        rho = spearmanr(pred[idx], obs[idx]).statistic
        ranking[f"fold_{f}"] = {"rho": None if np.isnan(rho) else float(rho),
                                "n": int(len(idx))}
    rho_all = spearmanr(pred, obs).statistic
    ranking["all"] = {"rho": float(rho_all), "n": len(data)}

    # ---- PPMC task: score every PPMC sequence with the fitted model ------
    tab = load_ppmc()
    cache: dict[str, dict] = {}
    scores, lab_d, lab_any, feats = [], [], [], []
    import re as _re
    _std = _re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
    for row in tab:
        seq = row["sequence"]
        if not _std.match(seq):
            continue
        f = _features_row(seq, None, cache)
        d = {"features": f, "salt_mM": DEFAULT_SALT_MM}
        scores.append(condition_score(d, params, version))
        L = row["labels"]
        lab_d.append(bool(L["ppmc_driver_plus"]))
        lab_any.append(bool(L["ppmc_driver_plus"] or L["ppmc_client_plus"]))
        feats.append(f)
    scores = np.array(scores, dtype=float)
    lab_d = np.array(lab_d, dtype=int)
    lab_any = np.array(lab_any, dtype=int)
    # score direction: lower predicted csat => more likely driver
    score_driver = -scores
    ppmc_eval = {
        "n": len(lab_d),
        "n_drivers_D_plus": int(lab_d.sum()),
        "n_any_LLPS": int(lab_any.sum()),
        "auc_D_plus_vs_rest": _roc_auc(lab_d, score_driver),
        "auc_any_LLPS_vs_rest": _roc_auc(lab_any, score_driver),
    }

    # ---- baselines on the same task --------------------------------------
    def _col(key):
        return np.array([fe["length"] if key == "length" else fe[key] for fe in feats],
                        dtype=float)

    def _auc_best_sign(v):
        a = _roc_auc(lab_d, v)
        return _roc_auc(lab_d, -v) if (a < 0.5 and not np.isnan(a)) else a

    baselines_d = {
        "abs_ncpr": _auc_best_sign(np.abs(_col("ncpr"))),
        "fcr": _auc_best_sign(_col("fcr")),
        "f_arom": _auc_best_sign(_col("f_arom")),
        "kappa": _auc_best_sign(_col("kappa")),
        "scd_abs": _auc_best_sign(np.abs(_col("scd"))),
        "length_shorter": _roc_auc(lab_d, -_col("length")),
        "random_expected": 0.5,
    }

    # ---- baselines on csat ranking ---------------------------------------
    base_rank = {}
    for key in ("ncpr", "fcr", "f_arom", "kappa", "scd", "length"):
        v = np.array([d["features"]["length"] if key == "length"
                      else d["features"][key] for d in data], dtype=float)
        r = spearmanr(v, obs).statistic
        base_rank[key] = {"rho": None if np.isnan(r) else float(r)}

    out = {
        "csat_ranking_model": ranking,
        "csat_ranking_descriptors": base_rank,
        "ppmc_classification_model": ppmc_eval,
        "baselines_ppmc_D_plus": baselines_d,
        "notes": [
            "Model score = predicted log10 csat (uM) from WP-B2c fit "
            "(data/processed/calibration_result.json), salt imputed "
            f"{DEFAULT_SALT_MM} mM for PPMC.",
            "AUC orientation: score = -log10 csat (lower csat = more likely driver).",
            "Csat ranking is bound-based (positive rows give only an upper bound).",
        ],
    }
    os.makedirs(PROCESSED, exist_ok=True)
    with open(os.path.join(PROCESSED, "evaluation.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    res = run()
    print(json.dumps(res["ppmc_classification_model"], indent=2))
    print("baselines:", json.dumps(res["baselines_ppmc_D_plus"], indent=2))
    print("ranking model:", json.dumps(res["csat_ranking_model"], indent=2))
