"""LLPSDB condition-level quantitative table (WP-B2a/b).

The LLPS.xls sheets record, per EXPERIMENT, the protein construct sequence,
its tested concentration (sometimes a range), salt, temperature, and the
observed outcome (droplets Yes/No). Single-protein positive/negative rows
together give genuine per-construct Csat BOUNDS:

    positive row at c   ->  csat <= c
    negative row at c   ->  csat >  c

which is exactly the inequality shape a semi-empirical chi/N_eff
calibration should fit (bracket loss), instead of point-regression on
scraped anecdotes.

Also parses explicit saturation-concentration sentences from PhaSePro
("... saturation concentration ... is 2 uM in 75 mM KCl") as a small
high-precision Csat point set.

Run: PYTHONPATH=src python3 -m darkmatter.data.conditions
"""
from __future__ import annotations

import io
import json
import os
import re
import zipfile

import pandas as pd

from darkmatter.data.ingest import _clean_seq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RAW = os.path.join(ROOT, "data", "raw")
PROCESSED = os.path.join(ROOT, "data", "processed")

AA_MASS_DA = 110.0  # mean residue mass (documented approximation for IDPs)

_NUM = r"(\d+(?:[.,]\d+)?)"
# NOTE: longest-first unit alternation so 'µM' is matched as ONE token,
# never split into a spurious 'µ' prefix + 'M' (the original bug).
_UNIT = r"(?:µg/ml|μg/ml|ug/ml|mg/ml|g/ml|µM|μM|uM|mM|M)"
_RANGE_RE = re.compile(rf"^{_NUM}\s*(?:-|–|to)\s*{_NUM}\s*({_UNIT})\b", re.I)
_SINGLE_RE = re.compile(rf"^{_NUM}\s*({_UNIT})\b", re.I)
_SALT_RE = re.compile(rf"{_NUM}\s*(mM|M)\s*(.+?)?$", re.I)


def _unit_to_uvM(value: float, unit: str, seq_len: int | None) -> float | None:
    """Convert a (value, unit) concentration to micromolar (protein monomer
    when mg/ml is resolvable via sequence length; None when not convertible)."""
    u = unit.strip().lower()
    if u == "m":       # mol/L
        return value * 1e6
    if u == "mm":
        return value * 1e3
    if u in ("µm", "μm", "um"):
        return value
    # mass-per-volume -> µM needs molecular weight
    if u == "mg/ml":
        mg_per_ml = value
    elif u in ("µg/ml", "μg/ml", "ug/ml"):
        mg_per_ml = value / 1000.0
    elif u == "g/ml":
        mg_per_ml = value * 1000.0
    else:
        return None
    if not seq_len:
        return None
    mw_g_per_mol = AA_MASS_DA * seq_len
    return mg_per_ml / mw_g_per_mol * 1e6  # (g/L) / (g/mol) -> mol/L -> µM


def parse_concentration(text: str, seq_len: int | None = None):
    """Parse 'Solute concentration' cell -> (c_uv_M or None, c_raw_max_uvM_or_None,
    kind) where kind in {'point','range','multi','unknown'}. c is the single
    tested point; for ranges we return the MAX as the conservative bound
    (a range row only certifies csat <= max for positives)."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None, None, "unknown"
    t = str(text).strip()
    m = _RANGE_RE.match(t)
    if m:
        v_max = float(m.group(2).replace(",", "."))
        conv = _unit_to_uvM(v_max, m.group(3), seq_len)
        return conv, conv, "range"
    m = _SINGLE_RE.match(t)
    if m:
        v = float(m.group(1).replace(",", "."))
        conv = _unit_to_uvM(v, m.group(2), seq_len)
        return conv, conv, "point"
    # multi-component or narrative ("2.5 uM [A]; 0.3 uM [B]") -> first number
    m = re.search(rf"{_NUM}\s*({_UNIT})\b", t, re.I)
    if m:
        v = float(m.group(1).replace(",", "."))
        conv = _unit_to_uvM(v, m.group(2), seq_len)
        return conv, conv, "multi"
    return None, None, "unknown"


def parse_salt_mM(text: str) -> float | None:
    """Salt concentration in mM (any salt; documented mono-valent caveat).
    'M' -> mol/L * 1000. Returns None for narrative or absent entries."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    t = str(text).strip()
    if not t or t in ("-", "NaN"):
        return None
    m = _SALT_RE.search(t)
    if m:
        v = float(m.group(1).replace(",", "."))
        # group(2) is 'mM' or 'M' only (regex is case-insensitive; compare
        # case-insensitively against 'mM'):
        return v * (1.0 if m.group(2).lower() == "mm" else 1000.0)
    return None


def parse_temperature_C(text: str) -> float | None:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    t = str(text).strip().lower()
    if t in ("-", "rt", "room temp", "room temperature", "nan", ""):
        return 25.0  # documented convention: RT -> 25 C
    if t.endswith("°c") or t.endswith("c"):
        try:
            return float(re.match(r"(\d+(?:\.\d+)?)", t).group(1))
        except (ValueError, AttributeError):
            return None
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", t):
        # values like "20" only
        return None
    try:
        return float(t)
    except ValueError:
        return None


# --- tables ------------------------------------------------------------------

def _read_sheet(zname: str, sheet: str) -> pd.DataFrame:
    with zipfile.ZipFile(os.path.join(RAW, zname + ".zip")) as zf:
        data = pd.read_excel(io.BytesIO(zf.read(f"{zname}/{sheet}")), sheet_name=None)
    return list(data.values())[0]


def load_single_protein_conditions() -> list[dict]:
    """All single-protein condition rows from the four Unambiguous + Ambiguous
    zips; Ambiguous zips have no conditions usable ("-" semantics), so we
    include them but concentration=None. Outcome read from the right column
    of each schema."""
    rows = []
    jobs = [
        ("Phase_separation_Unambiguous", "LLPS.xls", True, "Phase separation"),
        ("No_phase_separation_Unambiguous", "LLPS.xls", False, "Phase separation"),
        ("Phase_diagram_Unambiguous", "LLPS.xls", True, "Phase diagram"),
    ]
    for zname, sheet, positive, outcome_col in jobs:
        df = _read_sheet(zname, sheet)
        df.columns = [str(c).strip() for c in df.columns]
        if "Components type" not in df.columns:
            continue
        for _, r in df.iterrows():
            if str(r.get("Components type", "")).strip().lower() != "protein(1)":
                continue
            seq = _clean_seq(r.get("Sequence"))
            if len(seq) < 10:
                continue
            outcome = str(r.get(outcome_col, "")).strip()
            conc, conc_max, kind = parse_concentration(
                r.get("Solute concentration"), seq_len=len(seq))
            rows.append({
                "psid": str(r.get("PSID", "")),
                "zname": zname,
                "positive": bool(positive),
                "outcome": outcome,
                "sequence": seq,
                "construct": str(r.get("Protein name", "")),
                "uniprot": str(r.get("Protein ID", "")),
                "conc_uvM": conc,
                "conc_max_uvM": conc_max,
                "conc_kind": kind,
                "salt_mM": parse_salt_mM(str(r.get("Salt concentration", ""))),
                "temp_C": parse_temperature_C(str(r.get("Temperature", ""))),
                "mutation": str(r.get("Mutation", "")),
                "source": "LLPSDB",
            })
    return rows


def load_phasepro_csat_points() -> list[dict]:
    """High-precision Csat sentences from PhaSePro experiment descriptions:
    '... saturation concentration ... is X uM ...'. Only sentences with the
    explicit phrase are used (no bare uM scraping)."""
    import csv
    path = os.path.join(RAW, "phasepro_full.json")
    with open(path) as fh:
        db = json.load(fh)
    out = []
    sent_re = re.compile(r"[^.]*(?:saturation concentration|csat|c_?sat)[^.]*\.", re.I)
    for acc, e in db.items():
        seq = _clean_seq(e.get("sequence"))
        blob = " ".join(str(e.get(k, "")) for k in ("experiment_llps", "determinants", "description"))
        for m in sent_re.finditer(blob):
            s = m.group(0)
            sent = s.replace("µ", "u").replace("μ", "u")
            nm = re.search(rf"{_NUM}\s*(u)?([mM])\b", sent)
            if not nm:
                continue
            val = float(nm.group(1).replace(",", "."))
            unit = ((nm.group(2) or "") + nm.group(3)).lower()
            if unit == "mm":
                val *= 1000.0
            # only accept sentences that actually assert a csat value
            if not re.search(r"(is|are|about|approximately|of)\s+(approximately\s+)?\d", s, re.I):
                continue
            if "csat" not in s.lower() and "saturation concentration" not in s.lower():
                continue
            out.append({
                "source": "PhaSePro",
                "source_id": acc,
                "gene": e.get("gene") or e.get("common_name", ""),
                "sequence": seq,
                "csat_uvM": val,
                "sentence": s.strip()[:300],
            })
    return out


def build_csat_brackets() -> dict:
    """Group condition rows by exact sequence; compute per-construct
    Csat bracket [lower, upper) from inequalities:
       positives: csat <= c ; negatives: csat > c."""
    rows = load_single_protein_conditions()
    groups: dict[str, dict] = {}
    for r in rows:
        g = groups.setdefault(r["sequence"], {
            "sequence": r["sequence"], "construct": r["construct"],
            "uniprot": r["uniprot"], "upper_uvM": None, "lower_uvM": None,
            "n_pos": 0, "n_neg": 0, "rows": [],
        })
        c = r["conc_uvM"]
        if r["positive"]:
            g["n_pos"] += 1
            if c is not None:
                g["upper_uvM"] = c if g["upper_uvM"] is None else min(g["upper_uvM"], c)
        else:
            g["n_neg"] += 1
            if c is not None:
                g["lower_uvM"] = c if g["lower_uvM"] is None else max(g["lower_uvM"], c)
        g["rows"].append(r["psid"])
    brackets = []
    for g in groups.values():
        low, up = g["lower_uvM"], g["upper_uvM"]
        consistent = (low is None) or (up is None) or (low < up)
        brackets.append({
            **{k: g[k] for k in ("sequence", "construct", "uniprot",
                                 "n_pos", "n_neg")},
            "csat_lower_uvM": low,
            "csat_upper_uvM": up,
            "consistent": bool(consistent),
        })
    out = {
        "n_condition_rows": len(rows),
        "n_constructs": len(brackets),
        "n_csat_bounded": sum(1 for b in brackets
                              if b["csat_lower_uvM"] is not None or b["csat_upper_uvM"] is not None),
        "n_inconsistent": sum(1 for b in brackets if not b["consistent"]),
        "brackets": brackets,
        "phasepro_points": load_phasepro_csat_points(),
    }
    os.makedirs(PROCESSED, exist_ok=True)
    with open(os.path.join(PROCESSED, "llpsdb_conditions.json"), "w") as fh:
        json.dump(out, fh)
    return out


if __name__ == "__main__":
    res = build_csat_brackets()
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("brackets", "phasepro_points")}, indent=2))
    print("PhaSePro csat points:", len(res["phasepro_points"]))
    for b in res["brackets"][:8]:
        lo, hi = b["csat_lower_uvM"], b["csat_upper_uvM"]
        fmt = lambda x: "-" if x is None else f"{x:.3g}"
        print(f"  {b['construct'][:28]:28s} pos={b['n_pos']:3d} neg={b['n_neg']:3d} "
              f"csat in ({fmt(lo)}, {fmt(hi)}] uM consistent={b['consistent']}")
