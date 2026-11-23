"""WP-B2c: sequence-calibrated Flory-Huggins chi / N_eff model.

Model (interpretable, semi-empirical):
    chi(seq, salt) = th0 + th1*ncpr^2 + th2*|scd|_norm + th3*fcr
                     + th4*f_arom + th5*kappa + th6*f_minus*log10(1+salt_mM)
    log10 N_eff    = g0 + g1*log10(N)          (global blob-length law)
    csat_pred      = phi_dilute(chi, N_eff) / (K_bead * N_eff)   -> uM

    phi_dilute is the dilute binodal root of the FH free energy, read from a
    precomputed bilinear log10-phi grid (the exact solver lives in
    darkmatter.physics.flory_huggins).

Fit: bracket loss on LLPSDB condition rows
    positive row at c   ->  log10 csat <= log10 c      (upper bound)
    negative row at c   ->  log10 csat >  log10 c      (lower bound)
plus squared residuals on the (small) PhaSePro explicit-csat point set.
All residuals clamp at HINGE cap (the inequality direction is one-sided).

Cross-validation: folds come from data/processed/splits.json
(cluster-safe: near-identical sequences never straddle train/test).

Run: PYTHONPATH=src python3 -m darkmatter.physics.calibration
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from darkmatter.data.conditions import (
    build_csat_brackets,
    load_single_protein_conditions,
)
from darkmatter.physics.charge import charge_statistics
from darkmatter.physics.flory_huggins import csat

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PROCESSED = os.path.join(ROOT, "data", "processed")

# documented imputation: rows with unrecorded salt -> physiologic ~150 mM
DEFAULT_SALT_MM = 150.0

# ---- FH dilute-root grid ----------------------------------------------------
PHI_MIN_LOG10 = -14.0   # below this csat is experimentally meaningless (< fM)
PHI_MAX_LOG10 = -0.2    # dense-branch ceiling


class PhiGrid:
    """Bilinear log10(phi_dilute) lookup over (chi, log10 N_eff)."""

    def __init__(self, chi_lo: float = 0.8, chi_hi: float = 5.0, n_chi: int = 85,
                 logn_lo: float = 0.7, logn_hi: float = 2.45, n_n: int = 36):
        self.chi = np.linspace(chi_lo, chi_hi, n_chi)
        self.logn = np.linspace(logn_lo, logn_hi, n_n)  # log10 N_eff
        # grid[row=n_idx, col=chi_idx] of log10 phi (NaN = no two-phase regime)
        self.lphi = np.full((n_n, n_chi), np.nan)
        for i, ln in enumerate(self.logn):
            n_eff = 10.0 ** ln
            for j, ch in enumerate(self.chi):
                phi = csat(min(ch, 0.999 * 5.0), n_eff)
                if phi is not None:
                    self.lphi[i, j] = np.log10(phi)

    def log10_phi(self, chi: float, log10_n: float) -> float:
        """Interpolated log10 phi_dilute; +1.0 above PHI_MAX_LOG10 if no
        two-phase regime (interior one-phase => effectively infinite csat)."""
        i = np.clip(np.searchsorted(self.logn, log10_n) - 1, 0, len(self.logn) - 2)
        j = np.clip(np.searchsorted(self.chi, chi) - 1, 0, len(self.chi) - 2)
        ln1, ln2 = self.logn[i], self.logn[i + 1]
        c1, c2 = self.chi[j], self.chi[j + 1]
        tn = (log10_n - ln1) / (ln2 - ln1) if ln2 > ln1 else 0.0
        tc = (chi - c1) / (c2 - c1) if c2 > c1 else 0.0
        v00, v01 = self.lphi[i, j], self.lphi[i, j + 1]
        v10, v11 = self.lphi[i + 1, j], self.lphi[i + 1, j + 1]
        corners = [v00, v01, v10, v11]
        if np.isnan(v00) and np.isnan(v01) and np.isnan(v10) and np.isnan(v11):
            return PHI_MAX_LOG10 + 1.0  # interior one-phase
        # fill NaN corners by nearest valid neighbour (edge of coexistence)
        vals = []
        for k, v in enumerate(corners):
            di, dj = divmod(k, 2)
            if np.isnan(v):
                vi = i + di - (1 if np.isnan(self.lphi[i + di, j + dj]) else 0)
                vj = j + dj - (1 if np.isnan(self.lphi[i + di, j + dj]) else 0)
                vi = int(np.clip(vi, 0, self.lphi.shape[0] - 1))
                vj = int(np.clip(vj, 0, self.lphi.shape[1] - 1))
                v = self.lphi[vi, vj]
                if np.isnan(v):
                    v = PHI_MAX_LOG10
                corners[k] = v
            vals.append(corners[k])
        v00, v01, v10, v11 = vals
        return float((1 - tn) * ((1 - tc) * v00 + tc * v01)
                     + tn * ((1 - tc) * v10 + tc * v11))


# ---- features ---------------------------------------------------------------

def _features_row(seq: str, salt_mM: float | None, cache: dict) -> dict:
    f = cache.get(seq)
    if f is None:
        try:
            st = charge_statistics(seq)
        except ZeroDivisionError:
            # uncharged sequence (FCR=0): delta_max=0 -> kappa undefined; 0
            st = charge_statistics.__wrapped__(seq) if hasattr(charge_statistics, "__wrapped__") else None
            st = {"length": len(seq), "ncpr": 0.0, "fcr": 0.0, "f_plus": 0.0,
                  "f_minus": 0.0, "sigma": 0.0, "kappa": 0.0, "scd": 0.0,
                  "delta": 0.0, "delta_max": 0.0}
        n = st["length"]
        f = {
            "length": n,
            "ncpr": st["ncpr"],
            "fcr": st["fcr"],
            "f_plus": st["f_plus"],
            "f_minus": st["f_minus"],
            "kappa": st["kappa"],
            "scd": st["scd"],
            "f_arom": sum(seq.count(a) for a in "WFY") / n,
            # WP-H1 extended descriptors
            "hydro": _mean_hydropathy(seq),
            "f_sticker": sum(seq.count(a) for a in "WYFR") / n,
            "f_R": seq.count("R") / n,
            "f_Y": seq.count("Y") / n,
            "f_PG": sum(seq.count(a) for a in "PG") / n,
        }
        cache[seq] = f
    return f


# Kyte-Doolittle scale (1982); mean hydropathy as sticker-chemistry covariate
_KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4,
           H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8,
           T=-0.7, W=-0.9, Y=-1.3, V=4.2)


def _mean_hydropathy(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(_KD.get(a, 0.0) for a in seq) / len(seq)


def assemble_dataset(grid: PhiGrid | None = None):
    """Row-level dataset for bracket fitting."""
    rows = load_single_protein_conditions()
    # cluster-safe fold lookup
    fold_of: dict[str, str] = {}
    spath = os.path.join(PROCESSED, "splits.json")
    if os.path.exists(spath):
        sp = json.load(open(spath))
        for rec in sp["records"]:
            fold_of[rec["sequence"]] = rec["fold"]
    cache: dict[str, dict] = {}
    data = []
    import re as _re
    _std = _re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
    for r in rows:
        seq = r["sequence"]
        if not _std.match(seq):
            continue  # host-cell tags / residue annotations in construct cells
        f = _features_row(seq, None, cache)
        salt = r["salt_mM"] if r["salt_mM"] is not None else DEFAULT_SALT_MM
        c = r["conc_uvM"]
        if c is None or c <= 0:
            continue
        data.append({
            "sequence": seq,
            "psid": r["psid"],
            "out_fold": fold_of.get(seq, "unassigned"),
            "is_positive": bool(r["positive"]),
            "c_uvM": c,
            "salt_mM": salt,
            "features": f,
        })
    # PhaSePro explicit csat points (high precision)
    from darkmatter.data.conditions import load_phasepro_csat_points
    for p in load_phasepro_csat_points():
        seq = p["sequence"]
        if not _std.match(seq):
            continue
        f = _features_row(seq, None, cache)
        data.append({
            "sequence": seq,
            "psid": p["source_id"],
            "out_fold": fold_of.get(seq, "unassigned"),
            "is_positive": True,
            "is_point": True,
            "c_uvM": p["csat_uvM"],
            "salt_mM": DEFAULT_SALT_MM,
            "features": f,
        })
    return data, fold_of


# ---- model ------------------------------------------------------------------

def chi_params_names(version: str = "v1") -> list[str]:
    base = ["th0", "th1_ncpr2", "th2_scdnorm", "th3_fcr", "th4_farom",
            "th5_kappa", "th6_salt"]
    if version == "v2":
        base += ["th7_hydro", "th8_fsticker"]
    return base


def chi_of(d: dict, theta, version: str = "v1") -> float:
    f = d["features"]
    chi = (theta[0] + theta[1] * f["ncpr"] ** 2 + theta[2] * min(abs(f["scd"]) / 3.0, 2.0)
           + theta[3] * f["fcr"] + theta[4] * f["f_arom"] + theta[5] * f["kappa"]
           + theta[6] * f["f_minus"] * np.log10(1.0 + d["salt_mM"]))
    if version == "v2":
        # normalised: hydro/3.5 ~ [-1.3, 1.1]; f_sticker/0.12 ~ [0, ~2]
        chi = chi + theta[7] * (f["hydro"] / 3.5) + theta[8] * (f["f_sticker"] / 0.12)
    return chi


def log_neff_of(d: dict, g, version: str = "v1") -> float:
    ln = g[0] + g[1] * np.log10(d["features"]["length"])
    if version == "v2":
        # per-sequence blob length: stickers shrink blobs, PG spacers grow them
        ln = ln + g[2] * (d["features"]["f_sticker"] / 0.12) + g[3] * (d["features"]["f_PG"] / 0.35)
    return ln


def model_vector(d: dict, theta, g, logK, version: str = "v1") -> np.ndarray:
    """Predicted log10 csat (uM) per row."""
    chi = chi_of(d, theta, version)
    logn = log_neff_of(d, g, version)
    lphi = grid_log10_phi(chi, logn)
    return lphi - logK - logn  # log10 phi - log10(K*Neff)


_grid_cache: PhiGrid | None = None


def grid_log10_phi(chi: float | np.ndarray, log10_n: float | np.ndarray):
    global _grid_cache
    if _grid_cache is None:
        _grid_cache = PhiGrid()
    g = _grid_cache
    chi = np.atleast_1d(np.asarray(chi, dtype=float))
    logn = np.atleast_1d(np.asarray(log10_n, dtype=float))
    out = np.empty(chi.shape)
    for k in range(chi.size):
        out.flat[k] = g.log10_phi(float(chi.flat[k]), float(logn.flat[k]))
    return out


def n_params(version: str) -> int:
    return {"v1": 10, "v2": 14}[version]


def residual_elements(d: dict, pred: float, obs: float) -> float:
    if d.get("is_point"):
        return pred - obs            # squared point residual
    if d["is_positive"]:             # csat_pred <= c
        return max(0.0, pred - obs)
    return max(0.0, obs - pred)      # csat_pred > c


def residuals(params, data, idx_train, version: str = "v1") -> np.ndarray:
    np_th = 9 if version == "v2" else 7
    ng = 4 if version == "v2" else 2
    theta, g, logK = params[:np_th], params[np_th:np_th + ng], params[-1]
    res = []
    for k in idx_train:
        d = data[k]
        pred = float(model_vector(d, theta, g, logK, version)[0])
        obs = float(np.log10(d["c_uvM"]))
        res.append(residual_elements(d, pred, obs))
    return np.array(res)


_V1_INIT = [3.0, -1.5, -0.4, 1.5, 0.5, 0.8, 0.0, 0.45, 0.2, -7.0]
_V1_LO = [-10.0, -10.0, -5.0, -10.0, -3.0, -3.0, -2.0, 0.0, -0.9, -9.0]
_V1_HI = [10.0, 10.0, 5.0, 10.0, 3.0, 3.0, 6.0, 1.5, 0.9, -5.0]
_V2_INIT = [3.0, -1.5, -0.4, 1.5, 0.5, 0.8, 0.0, 0.0, 0.0,
            0.45, 0.2, 0.0, 0.0, -7.0]
_V2_LO = [-10.0, -10.0, -5.0, -10.0, -3.0, -3.0, -2.0, -6.0, -6.0,
          0.0, -0.9, -4.0, -4.0, -9.0]
_V2_HI = [10.0, 10.0, 5.0, 10.0, 3.0, 3.0, 6.0, 6.0, 6.0,
          1.5, 0.9, 4.0, 4.0, -5.0]


def fit(data, idx_train, version: str = "v1", x0=None):
    init, lo, hi = {"v1": (_V1_INIT, _V1_LO, _V1_HI),
                    "v2": (_V2_INIT, _V2_LO, _V2_HI)}[version]
    if x0 is None:
        x0 = np.array(init)
    out = least_squares(residuals, x0, bounds=(np.array(lo), np.array(hi)),
                        args=(data, idx_train, version), xtol=1e-10, ftol=1e-10)
    return out


def unpack(params, version: str):
    if version not in ("v1", "v2"):
        raise ValueError(f"unknown model version {version!r}")
    np_th = 9 if version == "v2" else 7
    ng = 4 if version == "v2" else 2
    return params[:np_th], params[np_th:np_th + ng], params[-1]


def condition_score(d: dict, params: np.ndarray, version: str = "v1") -> float:
    """Predicted log10 csat (uM) for one dataset row (dict with 'features'
    and 'salt_mM'). Larger = more concentrated needed = weaker driver."""
    theta, g, logK = unpack(params, version)
    return float(model_vector(d, theta, g, logK, version)[0])


def predict_log10_csat(data_index, params, data, version: str = "v1") -> float:
    theta, g, logK = unpack(params, version)
    return float(model_vector(data[data_index], theta, g, logK, version)[0])


# ---- main -------------------------------------------------------------------

def _rank_metrics(data, params, idx, version: str = "v1") -> dict:
    """Spearman rho of predicted log10 csat vs the tested bound."""
    from scipy.stats import spearmanr
    preds, obs = [], []
    for k in idx:
        preds.append(predict_log10_csat(k, params, data, version))
        obs.append(np.log10(data[k]["c_uvM"]))
    preds, obs = np.array(preds), np.array(obs)
    if len(preds) < 3:
        return {"spearman_rho": None, "n": len(preds)}
    rho = spearmanr(preds, obs).statistic
    return {"spearman_rho": None if np.isnan(rho) else float(rho), "n": len(preds)}


def crossvalidate(data, folds, version: str, verbose: bool = True) -> dict:
    out_folds = {}
    rhos = []
    for f in folds:
        tr = [k for k, d in enumerate(data) if d["out_fold"] != f]
        te = [k for k, d in enumerate(data) if d["out_fold"] == f]
        out = fit(data, tr, version)
        met_te = _rank_metrics(data, out.x, te, version)
        out_folds[f] = {"n_train": len(tr), "n_test": len(te),
                        "spearman_test": met_te["spearman_rho"],
                        "n": met_te["n"], "params": out.x.tolist(),
                        "cost": float(out.cost)}
        rhos.append(met_te["spearman_rho"])
        if verbose:
            print(f"[{version}] fold {f}: test rho={met_te['spearman_rho']} "
                  f"n={met_te['n']}")
    good = [r for r in rhos if r is not None]
    out_folds["mean_rho"] = float(np.mean(good)) if good else None
    return out_folds


def main():
    data, _ = assemble_dataset()
    print(f"rows: {len(data)}")
    folds = sorted({d["out_fold"] for d in data if d["out_fold"] != "unassigned"})
    print("folds:", folds, "+ unassigned:",
          sum(1 for d in data if d["out_fold"] == "unassigned"))

    results = {"n_rows": len(data), "features_doc": {
        "sigma_imputed_salt_mM": DEFAULT_SALT_MM,
        "scd_norm": "min(|SCD|/3, 2)",
        "chi_terms": chi_params_names("v2"),
        "N_eff_laws": {
            "v1": "log10 Neff = g0 + g1*log10(N)",
            "v2": "+ g2*(f_sticker/0.12) + g3*(f_PG/0.35)",
        },
        "csat_uM": "phi_dilute/(K_bead*Neff)",
    }}

    # model selection: cluster-safe 5-fold CV for each version
    cv = {}
    for version in ("v1", "v2"):
        cv[version] = crossvalidate(data, folds, version)
        print(f"[{version}] mean test rho = {cv[version]['mean_rho']}")
    results["cv"] = {
        v: {k: x for k, x in cv[v].items() if k != "params"}
        for v in cv
    }
    # parsimony rule: keep the simpler v1 unless v2 gains >0.02 mean rho
    m1, m2 = cv["v1"]["mean_rho"], cv["v2"]["mean_rho"]
    best = "v2" if (m2 is not None and m1 is not None and m2 - m1 > 0.02) else "v1"
    results["model_version"] = best
    results["selection_rule"] = (
        "prefer simpler v1 unless v2 improves mean CV rho by >0.02 "
        f"(v1={m1}, v2={m2} -> {best})")
    print("best model version:", best)

    # final fit of BOTH versions on all rows (best is 'final_fit')
    all_idx = list(range(len(data)))
    for version in ("v1", "v2"):
        out = fit(data, all_idx, version)
        theta, g, logK = unpack(out.x, version)
        entry = {
            "params": out.x.tolist(),
            "theta": dict(zip(chi_params_names(version), theta.tolist())),
            "g": g.tolist(),
            "log10_K_bead": float(logK),
            "cost": float(out.cost),
            "spearman_all": _rank_metrics(data, out.x, all_idx, version),
        }
        key = "final_fit" if version == best else f"final_fit_{version}"
        results[key] = entry
    print(json.dumps(results["final_fit"], indent=2)[:1500])

    os.makedirs(PROCESSED, exist_ok=True)
    with open(os.path.join(PROCESSED, "calibration_result.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("wrote data/processed/calibration_result.json")


if __name__ == "__main__":
    main()
