"""WP-D2: mutation-level validation.

For each PhaSePro disease mutation (FUS, TARDBP, hnRNPA1, ...) we apply the
substitution to the wild-type driver-region sequence, run the calibrated
chi/N_eff csat model (WP-B2c fit, no mutation-specific refitting), and report
the predicted change in saturation concentration:

    d_log10_csat = log10 csat(mut) - log10 csat(wt)

Positive => mutation raises csat (weakens LLPS); negative => lowers csat.
Sign agreement is computed where PhaSePro direction is explicit
('increases'/'decreases' LLPS propensity) and where residue position falls
inside the annotated driver region. NOTE: the semi-empirical chi model
contains mutations only through charge descriptors (+local kappa/SCD shifts),
so this is a test of directional charge-driven response, NOT of aromatic
stickers or structure (documented limitation).

Run: PYTHONPATH=src python3 -m darkmatter.physics.mutation_eval
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from darkmatter.data.ingest import load_phasepro
from darkmatter.data.mutations import apply_mutation, MUT_RE
from darkmatter.physics.calibration import DEFAULT_SALT_MM, condition_score, _features_row

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PROCESSED = os.path.join(ROOT, "data", "processed")


def main():
    cal = json.load(open(os.path.join(PROCESSED, "calibration_result.json")))
    params = np.array(cal["final_fit"]["params"])
    version = cal.get("model_version", "v1")

    cache: dict[str, dict] = {}
    rows = []
    agg = {g: [] for g in set()}  # gene -> dlog10
    for rec in load_phasepro():
        wt = rec["sequence"]
        nmuts = 0
        for m in rec.get("disease_mutations", []):
            mv = m["mutation"].upper().replace("P.", "").replace("p.", "")
            match = MUT_RE.match(mv)
            if not match:
                continue
            wt_res, pos_s, mut_res = match.groups()
            pos = int(pos_s)
            if wt_res == mut_res:
                continue
            mut_seq = list(wt)
            if not apply_mutation(mut_seq, wt_res, pos, mut_res):
                continue
            mut_seq = "".join(mut_seq)
            nmuts += 1
            dw = {"features": _features_row(wt, None, cache),
                  "salt_mM": DEFAULT_SALT_MM}
            dm = {"features": _features_row(mut_seq, None, cache),
                  "salt_mM": DEFAULT_SALT_MM}
            lwt = condition_score(dw, params)
            lmu = condition_score(dm, params)
            b = rec.get("driver_region", {}).get("boundaries", "")
            nums = re.findall(r"\d+", str(b))
            in_driver = None
            if len(nums) >= 2:
                lo, hi = sorted(int(v) for v in nums[:2])
                in_driver = 1 <= lo <= pos <= hi <= len(wt)
            rows.append({
                "uniprot": rec["source_id"],
                "gene": rec["gene"],
                "mutation": mv,
                "position": pos,
                "disease": m.get("disease", ""),
                "direction": m.get("direction", ""),
                "pmids": m.get("pmids", ""),
                "in_driver_region": in_driver,
                "log10_csat_wt": lwt,
                "log10_csat_mut": lmu,
                "d_log10_csat": lmu - lwt,
            })
            agg.setdefault(rec["gene"], []).append(lmu - lwt)
        if nmuts:
            print(f"{rec.get('gene', '?')} ({rec['source_id']}): {nmuts} mutations scored")

    # sign agreement where direction is explicit in PhaSePro annotation
    def _dir_sign(d: str) -> int | None:
        d = d.lower()
        if "increase" in d or "promot" in d or "enhanc" in d:
            return -1  # enhanced LLPS -> csat goes DOWN
        if "decrease" in d or "inhibit" in d or "impair" in d:
            return +1  # weakened LLPS -> csat goes UP
        return None

    scored = [r for r in rows if _dir_sign(r["direction"]) is not None]
    agree = sum(1 for r in scored
                if np.sign(r["d_log10_csat"] or 0) == _dir_sign(r["direction"]))
    n_in_driver = sum(1 for r in rows if r["in_driver_region"])
    agree_in_driver = sum(1 for r in rows
                          if r["in_driver_region"] and _dir_sign(r["direction"]) is not None
                          and np.sign(r["d_log10_csat"] or 0) == _dir_sign(r["direction"]))
    ndir = sum(1 for r in rows if _dir_sign(r["direction"]) is not None)
    out = {
        "source": "PhaSePro disease annotations; model from WP-B2c (no refit)",
        "n_mutations": len(rows),
        "n_with_explicit_direction": ndir,
        "sign_agreement_directional": agree,
        "sign_agreement_rate": (agree / ndir) if ndir else None,
        "n_in_driver_region": n_in_driver,
        "sign_agreement_in_driver":
            (agree_in_driver / sum(1 for r in rows if r["in_driver_region"]
                                   and _dir_sign(r["direction"]) is not None))
            if any(r["in_driver_region"] and _dir_sign(r["direction"]) is not None
                   for r in rows) else None,
        "per_gene_mean_d_log10_csat":
            {g: float(np.mean(v)) for g, v in agg.items()},
        "records": rows,
    }
    os.makedirs(PROCESSED, exist_ok=True)
    with open(os.path.join(PROCESSED, "mutation_validation.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
