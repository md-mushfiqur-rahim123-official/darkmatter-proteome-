"""Pathological-mutation benchmark extraction (WP-D1).

Builds data/processed/mutation_benchmark.json: for each clinically curated
few-residue substitution in PhaSePro-documented LLPS proteins (FUS P35637,
TDP-43 Q13148, hnRNPA1 P09651, plus any other entry with disease
annotations), compute the full descriptor set for the wild-type sequence,
the mutated sequence, and the mutated DRIVER REGION (when the mutation
falls inside the curated driver segment).

We report DESCRIPTOR SHIFTS (kappa, SCD, NCPR, FCR, sticker counts);
phase-boundary claims via chi/N_eff calibration are WP-B2 output and are
NOT asserted here (honest scope: no p-value theatre).

Run:  PYTHONPATH=src python3 -m darkmatter.data.mutations
"""
from __future__ import annotations

import json
import os
import re

from darkmatter.physics.charge import charge_statistics
from darkmatter.data.ingest import load_phasepro

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PROCESSED = os.path.join(ROOT, "data", "processed")

MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
TARGETS = {"FUS": "P35637", "TARDBP": "Q13148", "HNRNPA1": "P09651"}


def apply_mutation(seq: str, wt: str, pos: int, mut: str) -> bool:
    """1-based position; validate WT residue; returns True on success."""
    if pos < 1 or pos > len(seq):
        return False
    if seq[pos - 1] != wt:
        return False
    seq[pos - 1] = mut
    return True


def extract(dest: str | None = None) -> dict:
    records = load_phasepro()
    out = []
    for rec in records:
        wt = rec["sequence"]
        nmuts = 0
        for m in rec.get("disease_mutations", []):
            mv = m["mutation"].upper().replace("P.", "").replace("p.", "")
            match = MUT_RE.match(mv)
            if not match:
                continue
            wt_res, pos_s, mut_res = match.groups()
            pos = int(pos_s)
            if wt_res == mut_res:  # e.g. synonymous annotations
                continue
            mut_seq = list(wt)
            if not apply_mutation(mut_seq, wt_res, pos, mut_res):
                # annotation/reference mismatch: record, do not silently drop
                out.append({
                    "uniprot": rec["source_id"], "gene": rec["gene"],
                    "mutation": mv, "status": "reference_mismatch",
                    "wt_residue_observed": wt[pos - 1] if pos <= len(wt) else None,
                    "disease": m.get("disease", ""), "direction": m.get("direction", ""),
                })
                continue
            nmuts += 1
            entry = {
                "uniprot": rec["source_id"],
                "gene": rec["gene"],
                "mutation": mv,
                "position": pos,
                "status": "ok",
                "disease": m.get("disease", ""),
                "direction": m.get("direction", ""),
                "pmids": m.get("pmids", ""),
                "in_driver_region": None,
            }
            # driver-region containment if boundaries are 1-based ints
            b = rec.get("driver_region", {}).get("boundaries", "")
            nums = re.findall(r"\d+", str(b))
            if len(nums) >= 2:
                lo, hi = sorted(int(v) for v in nums[:2])
                if 1 <= lo <= hi <= len(wt):
                    entry["in_driver_region"] = lo <= pos <= hi
                    entry["driver_region_bounds"] = [lo, hi]
            wt_stats = charge_statistics(wt)
            mu_stats = charge_statistics("".join(mut_seq))
            entry["wt"] = {k: (v if isinstance(v, (int, float)) else v)
                           for k, v in wt_stats.items()}
            entry["mut"] = {k: (v if isinstance(v, (int, float)) else v)
                            for k, v in mu_stats.items()}
            entry["shifts"] = {k: round(mu_stats[k] - wt_stats[k], 12)
                               for k in ("kappa", "scd", "ncpr", "fcr",
                                         "f_plus", "f_minus")}
            # per-window delta on the local patch: charge statistics on a
            # +-15 residue window centred on the mutation (sequence context)
            lo = max(0, pos - 16)
            hi = min(len(wt), pos + 15)
            entry["local_wt"] = charge_statistics(wt[lo:hi])
            entry["local_mut"] = charge_statistics("".join(mut_seq[lo:hi]))
            out.append(entry)
        if rec["source_id"] in TARGETS.values():
            print(f"{rec['gene']} ({rec['source_id']}): {nmuts} applied mutations, "
                  f"len(wt)={len(wt)}")
    payload = {
        "source": "PhaSePro disease-annotation fields (parsed)",
        "n_records": len(out),
        "n_applied": sum(1 for r in out if r["status"] == "ok"),
        "n_mismatched": sum(1 for r in out if r["status"] == "reference_mismatch"),
        "records": out,
    }
    os.makedirs(PROCESSED, exist_ok=True)
    dest = dest or os.path.join(PROCESSED, "mutation_benchmark.json")
    with open(dest, "w") as fh:
        json.dump(payload, fh, indent=1)
    return payload


if __name__ == "__main__":
    payload = extract()
    print(json.dumps({k: v for k, v in payload.items() if k != "records"},
                     indent=2))
