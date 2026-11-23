"""WP-F1: publication figures (SVG/PNG under results/figures/).

Fig 1: FH phase diagram (binodal + spinodal) for calibrated N_eff.
Fig 2: predicted vs observed log10 csat on LLPSDB condition rows
       (bracket visualization: positives -> upper bound arrows).
Fig 3: salt dependence of predicted csat (chi salt term).
Fig 4: PPMC driver separation (score distributions).

Run: PYTHONPATH=src python3 scripts/make_figures.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from darkmatter.physics.flory_huggins import binodal, critical_point  # noqa: E402
from darkmatter.physics.calibration import assemble_dataset, condition_score  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED = os.path.join(ROOT, "data", "processed")
FIGDIR = os.path.join(ROOT, "results", "figures")


def fig1_phase_diagram(n_eff: float, ax=None):
    ax = ax or plt.gca()
    _, chi_c = critical_point(n_eff)
    chis = np.linspace(chi_c, 4.0, 120)
    b = binodal(chis, n_eff)
    pts = [r for r in b if r["coexistence"]]
    ax.semilogx([r["phi_dilute"] for r in pts], [r["chi"] for r in pts],
                "b-", label="binodal (common tangent)")
    ax.semilogx([r["phi_dense"] for r in pts], [r["chi"] for r in pts],
                "b-", color="tab:blue")
    spins = np.logspace(-6, -0.02, 400)
    ax.semilogx(spins, [1.0 / (n_eff * q) + 1.0 / (1 - q) for q in spins],
                "r--", label="spinodal")
    # csat upper axis example
    ax.set_xlim(1e-12, 1)
    ax.set_ylim(0.4, 5)
    ax.set_xlabel(r"volume fraction $\phi$ (dilute + dense branches)")
    ax.set_ylabel(r"$\chi$")
    ax.set_title(f"FH coexistence, $N_{{\\rm eff}}$={n_eff:.0f}")
    ax.legend(fontsize=8)
    return ax


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    cal = json.load(open(os.path.join(PROCESSED, "calibration_result.json")))
    p = cal["final_fit"]["params"]
    g = p[7:9]

    # Fig 1
    fig, ax = plt.subplots(figsize=(5, 4))
    fig1_phase_diagram(10 ** (g[0] + g[1] * np.log10(450)), ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig1_phase_diagram.png"), dpi=200)
    plt.close(fig)

    # Fig 2: predicted vs bracket bounds
    data, _ = assemble_dataset()
    pred = np.array([condition_score(d, p) for d in data])
    obs = np.array([np.log10(d["c_uvM"]) for d in data])
    pos = np.array([d["is_positive"] for d in data])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(obs[pos], pred[pos], s=6, alpha=0.4, color="tab:blue",
               label="positive ($c$ = upper bound)", rasterized=True)
    ax.scatter(obs[~pos], pred[~pos], s=6, alpha=0.4, color="tab:red",
               label="negative ($c$ = lower bound)", rasterized=True)
    lim = [min(obs.min(), pred.min()) - 1, max(obs.max(), pred.max()) + 1]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel(r"observed bound: $\log_{10} c_{\rm tested}$ ($\mu$M)")
    ax.set_ylabel(r"$\log_{10}\hat{c}_{\rm sat}$ ($\mu$M)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig2_pred_vs_observe_bounds.png"), dpi=200)
    plt.close(fig)

    # Fig 3: salt dependence for a FUS-like sequence
    from darkmatter.physics.charge import charge_statistics
    fus = next(d for d in data if d["sequence"].startswith("MASNDYTQ"))
    salts = np.logspace(0, 3, 60)
    preds = []
    for s in salts:
        d = {"features": fus["features"], "salt_mM": float(s)}
        preds.append(condition_score(d, p))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.semilogx(salts, preds, "g-")
    ax.set_xlabel("salt (mM, mono-valent equiv.)")
    ax.set_ylabel(r"$\log_{10}\hat{c}_{\rm sat}$ ($\mu$M)")
    ax.set_title("salt screening raises predicted $c_{\\rm sat}$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig3_salt_dependence.png"), dpi=200)
    plt.close(fig)

    # Fig 4: PPMC separation
    ev = json.load(open(os.path.join(PROCESSED, "evaluation.json")))
    print("evaluation summary already computed; AUCs:",
          ev["ppmc_classification_model"]["auc_D_plus_vs_rest"],
          ev["ppmc_classification_model"]["auc_any_LLPS_vs_rest"])
    print("figures written to", FIGDIR)


if __name__ == "__main__":
    main()
