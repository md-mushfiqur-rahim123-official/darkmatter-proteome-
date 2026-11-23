#!/usr/bin/env python3
"""Provenance-controlled re-harvest of all raw sources (idempotent).

Downloads every file recorded in data/PROVENANCE.md into data/raw/ ONLY if
the file (or its digest line in DIGESTS.txt) is missing — never overwrites
an existing raw snapshot. Skips gracefully with a WARN when a host is
unreachable so partial harvests stay reproducible.

Usage: python3 scripts/harvest.py [--include-ppmc]

The PPMC datasets.tsv lives in a separate git repo
(https://github.com/PPMC-lab/llps-datasets); re-harvest it via:
    gh repo clone PPMC-lab/llps-datasets <somewhere>
then copy datasets.tsv to data/raw/ppmc_datasets.tsv (the ingest module
has a fallback path). --include-ppmc copies it if present.
"""
import hashlib
import os
import shutil
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")

USER_AGENT = ("DarkMatterProteome-harvest/0.1 "
              "(research; contact: repo owner)")

SOURCES = {
    "phasepro_full.json":
        "https://phasepro.elte.hu/download_full.json",
    "drllps_LLPS.txt":
        "http://llps.biocuckoo.cn/download/LLPS.txt",
    "Phase_separation_Unambiguous.zip":
        "http://bio-comp.org.cn/llpsdbv2/download/Phase_separation_Unambiguous.zip",
    "Phase_separation_ambiguous.zip":
        "http://bio-comp.org.cn/llpsdbv2/download/Phase_separation_ambiguous.zip",
    "No_phase_separation_Unambiguous.zip":
        "http://bio-comp.org.cn/llpsdbv2/download/No_phase_separation_Unambiguous.zip",
    "No_phase_separation_ambiguous.zip":
        "http://bio-comp.org.cn/llpsdbv2/download/No_phase_separation_ambiguous.zip",
    "Phase_diagram_Unambiguous.zip":
        "http://bio-comp.org.cn/llpsdbv2/download/Phase_diagram_Unambiguous.zip",
    "Phase_diagram_ambiguous.zip":
        "http://bio-comp.org.cn/llpsdbv2/download/Phase_diagram_ambiguous.zip",
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_digests() -> dict[str, str]:
    path = os.path.join(RAW, "DIGESTS.txt")
    out = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#"):
                digest, _, name = line.partition("  ")
                out[name.strip()] = digest.strip()
    return out


def main() -> int:
    os.makedirs(RAW, exist_ok=True)
    digests = load_digests()
    rc = 0
    for name, url in SOURCES.items():
        dest = os.path.join(RAW, name)
        want = digests.get(name)
        if os.path.exists(dest) and (want is None or sha256(dest) == want):
            print(f"OK-EXISTS  {name}")
            continue
        if os.path.exists(dest) and want is not None:
            print(f"WARN      {name}: exists but digest mismatch; leaving "
                  f"raw snapshot untouched (immutable-snapshot rule)")
            rc = 1
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as fh:
                shutil.copyfileobj(resp, fh)
            got = sha256(dest)
            print(f"DOWNLOADED {name} sha256={got[:12]}...")
            if want is not None and got != want:
                print(f"WARN      {name}: digest differs from ledger "
                      f"(upstream changed?); remove file+update ledger or investigate")
                rc = 1
        except Exception as exc:  # network-safe failure
            print(f"WARN      {name}: harvest failed ({exc}); snapshot absent")
            if os.path.exists(dest):
                os.remove(dest)
            rc = 1

    if "--include-ppmc" in sys.argv:
        candidates = [
            "/workspace/github__PPMC-lab__llps-datasets/datasets.tsv",
            os.path.join(ROOT, "github__PPMC-lab__llps-datasets/datasets.tsv"),
            os.path.join(ROOT, "..", "github__PPMC-lab__llps-datasets/datasets.tsv"),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                shutil.copy(cand, os.path.join(RAW, "ppmc_datasets.tsv"))
                print(f"COPIED     ppmc_datasets.tsv from {cand}")
                break
        else:
            print("WARN ppmc datasets.tsv not found; clone "
                  "PPMC-lab/llps-datasets and rerun with --include-ppmc")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
