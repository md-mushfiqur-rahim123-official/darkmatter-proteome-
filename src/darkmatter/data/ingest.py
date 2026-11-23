"""Unified ingestion of all four LLPS ground-truth sources.

Sources (see data/PROVENANCE.md for provenance and digests):
  A. PhaSePro full JSON        -> curated LLPS drivers (121), driver-region
                                  boundaries, disease mutations with direction.
  B. DrLLPS seed list (TSV)    -> Scaffold (150, strong drivers), Regulator
                                  (987), Client (8,144) with sequences.
  C. PPMC confident datasets   -> Genome Biology 2025 benchmark:
                                  D+ (driver in >=3 sources), D-, C+, C-,
                                  and '-' (potential non-LLPS negatives).
  D. LLPSDB v2.0 (zip/xls)     -> condition-level tested observations:
                                  Phase_separation (positive), 
                                  No_phase_separation (TESTED negatives),
                                  Phase_diagram (quantitative calibration).

Output: data/processed/unified_proteins.json -- one record per UniProt
sequence, with per-source labels kept SEPARATE (no silent label merging):
qualities differ by source confidence. Run as a script:
    PYTHONPATH=src python -m darkmatter.data.ingest
"""
from __future__ import annotations

import csv
import io
import json
import os
import zipfile

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RAW = os.path.join(ROOT, "data", "raw")
PROCESSED = os.path.join(ROOT, "data", "processed")


def _clean_seq(v) -> str:
    """Cell values may be plain sequences or FASTA text where line breaks
    are newlines (Unambiguous zips) or ';' (Ambiguous zips), e.g.
    '>sp|P...|NAME ...;MGKKT...;ELIS...'. Split on both separators, drop
    header chunks, drop whitespace, uppercase."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    text = str(v)
    chunks = [c.strip() for ln in text.splitlines()
              for c in ln.split(";")]
    seq = "".join(c.replace(" ", "") for c in chunks
                  if c and not c.startswith(">"))
    return seq.upper()


def load_phasepro(path: str | None = None) -> list[dict]:
    path = path or os.path.join(RAW, "phasepro_full.json")
    with open(path) as fh:
        db = json.load(fh)
    out = []
    for acc, e in db.items():
        seq = _clean_seq(e.get("sequence", ""))
        if not seq or not e.get("in_vitro") and not e.get("in_vivo"):
            continue
        mutations = []
        if isinstance(e.get("disease"), str) and e["disease"] != "Not known.":
            for line in e["disease"].splitlines():
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 1 and parts[0] not in ("Not known.", ""):
                    mutations.append(
                        {"mutation": parts[0],
                         "direction": parts[4] if len(parts) > 4 else "",
                         "disease": parts[2] if len(parts) > 2 else "",
                         "pmids": parts[5] if len(parts) > 5 else ""}
                    )
        out.append({
            "source": "PhaSePro",
            "source_id": acc,
            "gene": e.get("gene", ""),
            "organism": e.get("organism", ""),
            "sequence": seq,
            "labels": {"phasepro_driver": True},
            "driver_region": {"segment": e.get("segment", ""),
                              "boundaries": e.get("boundaries", "")},
            "disease_mutations": mutations,
            "pmids": e.get("pmids", ""),
        })
    return out


def load_drllps(path: str | None = None) -> list[dict]:
    path = path or os.path.join(RAW, "drllps_LLPS.txt")
    with open(path, errors="replace") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    out = []
    for r in rows:
        seq = _clean_seq(r.get("Protein Sequence"))
        if not seq:
            continue
        ltype = r["LLPS Type"]
        out.append({
            "source": "DrLLPS",
            "source_id": r["UniProt ID"],
            "gene": r.get("Gene name", ""),
            "organism": r.get("Species", ""),
            "sequence": seq,
            "labels": {"drllps_type": ltype},  # Scaffold | Regulator | Client
            "condensate": r.get("Condensate", ""),
            "references": r.get("References", ""),
        })
    return out


def load_ppmc(path: str | None = None) -> list[dict]:
    """PPMC benchmark. Label decoding from scripts/generate_datasets.py:
    D+ = driver listed in >=3 source datasets; D- = <3; C+ = client in both
    client sources; C- = one; no D/C codes = potential non-LLPS negative."""
    path = path or os.path.join(RAW, "ppmc_datasets.tsv")
    if not os.path.exists(path):
        path = "/workspace/github__PPMC-lab__llps-datasets/datasets.tsv"
    with open(path, errors="replace") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    out = []
    for r in rows:
        codes = set(r["Datasets"].split(";")) - {"_"}
        out.append({
            "source": "PPMC",
            "source_id": r["UniProt.Acc"],
            "gene": r["Gene.Name"],
            "organism": "",  # PPMC ships no organism column
            "sequence": _clean_seq(r["Full.seq"]),
            "labels": {
                "ppmc_driver_plus": "D+" in codes,
                "ppmc_driver_minus": "D-" in codes,
                "ppmc_client_plus": "C+" in codes,
                "ppmc_client_minus": "C-" in codes,
                "ppmc_non_llps_negative": codes == set() or codes == {"_"},
            },
        })
    return out


_ZIP_SHEETS = {
    "Phase_separation_Unambiguous": "llpsdb_positive_unambiguous",
    "Phase_separation_ambiguous": "llpsdb_positive_ambiguous",
    "No_phase_separation_Unambiguous": "llpsdb_negative_tested_unambiguous",
    "No_phase_separation_ambiguous": "llpsdb_negative_tested_ambiguous",
    "Phase_diagram_Unambiguous": "llpsdb_phase_diagram_unambiguous",
    "Phase_diagram_ambiguous": "llpsdb_phase_diagram_ambiguous",
}


def load_llpsdb(rawdir: str | None = None) -> list[dict]:
    rawdir = rawdir or RAW
    out = []
    for zname, tag in _ZIP_SHEETS.items():
        zp = os.path.join(rawdir, zname + ".zip")
        if not os.path.exists(zp):
            continue
        zf = zipfile.ZipFile(zp)
        xls = [n for n in zf.namelist() if n.endswith(".xls")]
        if not xls:
            continue
        # LLPSDB ships two files joined by PSID: protein.xls (sequence)
        # and LLPS.xls (conditions + separation status). Join them explicitly.
        parts = {}
        for n in xls:
            first = next(iter(pd.read_excel(io.BytesIO(zf.read(n)), sheet_name=None).values()))
            parts[n.split("/")[-1]] = first
        protein = parts.get("protein.xls")
        llps = parts.get("LLPS.xls")
        if llps is None:  # fail loudly: LLPS.xls carries status + conditions
            raise ValueError(f"{zname}.zip: no LLPS.xls sheet found in {[n for n in xls]}")
        # LLPSDB v2.0 ships TWO different schemas:
        #   Unambiguous: LLPS.xls has PSID, 'Protein ID', Sequence, lowercase conds
        #   Ambiguous  : LLPS.xls has PSID, 'protID',     no sequence, Solute/Salt...
        # protein.xls is enrichment (UniprotID/Species), keyed on PID in both.
        # Correct join: LLPS protein-key  <->  protein.xls['PID']  (NOT PSID).
        llps_cols = {str(c).strip().lower(): c for c in llps.columns}
        join_key = llps_cols.get("protein id") or llps_cols.get("protid")
        df = llps.copy()
        if protein is not None and join_key is not None and "PID" in protein.columns:
            psub = protein.drop_duplicates(subset="PID")
            df = df.merge(psub, left_on=join_key, right_on="PID",
                          how="left", suffixes=("", "_p"))
        cols = {str(c).strip().lower(): c for c in df.columns}
        seqcol = next((cols[c] for c in ("sequence", "protein sequence") if c in cols), None)
        if seqcol is None:
            raise ValueError(f"{zname}.zip: no sequence column; got {list(df.columns)}")
        idcol = cols.get("psid") or cols.get("protein id") or cols.get("pid")
        for _, row in df.iterrows():
            seq = _clean_seq(row[seqcol])
            if len(seq) < 5:
                continue
            rec = {
                "source": "LLPSDB",
                "tag": tag,
                "source_id": str(row[idcol]) if idcol and pd.notna(row[idcol]) else "",
                "sequence": seq,
                "components_type": str(row[cols["components type"]]) if "components type" in cols and pd.notna(row[cols["components type"]]) else "",
                "protein_type": str(row[cols["protein type (n/d)"]]) if "protein type (n/d)" in cols and pd.notna(row[cols["protein type (n/d)"]]) else "",
            }
            if cols.get("uniprot id") and pd.notna(row.get(cols["uniprot id"])):
                rec["uniprot"] = str(row[cols["uniprot id"]])
            if cols.get("uniprotid") and pd.notna(row.get(cols["uniprotid"])):
                rec["uniprot"] = str(row[cols["uniprotid"]])
            ps_val = None
            for c in ("phase separation", "phase_separation"):
                if cols.get(c) and pd.notna(row.get(cols[c])):
                    ps_val = str(row[cols[c]])[:40]
                    break
            if ps_val is not None:
                rec["phase_separation"] = ps_val
            for cond in ("solute", "salt", "crowding", "buffer", "temperature", "ph"):
                if cond in cols:
                    rec[cond] = str(row[cols[cond]])[:80]
            out.append(rec)
    return out


def build_unified_table(dest: str | None = None) -> dict:
    os.makedirs(PROCESSED, exist_ok=True)
    sources = {
        "PhasePro": load_phasepro(),
        "DrLLPS": load_drllps(),
        "PPMC": load_ppmc(),
        "LLPSDB": load_llpsdb(),
    }
    summary = {k: {"records": len(v),
                   "unique_sequences": len({r["sequence"] for r in v if "sequence" in r})}
               for k, v in sources.items()}
    payload = {"summary": summary, "sources": sources}
    dest = dest or os.path.join(PROCESSED, "unified_raw_sources.json")
    with open(dest, "w") as fh:
        json.dump(payload, fh)
    return payload


if __name__ == "__main__":
    payload = build_unified_table()
    print(json.dumps(payload["summary"], indent=2))
