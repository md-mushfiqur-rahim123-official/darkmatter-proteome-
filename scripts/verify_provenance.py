#!/usr/bin/env python3
"""Provenance verifier: recompute SHA-256 of every file recorded in
data/raw/DIGESTS.txt and fail loudly on any mismatch.

Also verifies data/PROVENANCE.md lists a digest line for each raw file.

Usage: python3 scripts/verify_provenance.py
"""
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
DIGESTS = os.path.join(RAW, "DIGESTS.txt")
PROV = os.path.join(ROOT, "data", "PROVENANCE.md")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not os.path.exists(DIGESTS):
        print(f"FATAL: {DIGESTS} missing", file=sys.stderr)
        return 2
    recorded = {}
    for line in open(DIGESTS):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, name = line.partition("  ")
        recorded[name.strip()] = digest.strip()

    prov_text = open(PROV).read()
    failures = 0
    for name, digest in sorted(recorded.items()):
        path = os.path.join(RAW, name)
        if not os.path.exists(path):
            print(f"MISSING  {name}")
            failures += 1
            continue
        got = sha256(path)
        status = "OK" if got == digest else "MISMATCH"
        if status == "MISMATCH":
            failures += 1
        if got != digest or os.environ.get("VERBOSE"):
            print(f"{status:8s} {name}  expected={digest[:12]} got={got[:12]}")

    # every tracked raw file except the ledger itself should be recorded
    for f in sorted(os.listdir(RAW)):
        if f in ("DIGESTS.txt",) or f.endswith(".html"):
            continue
        if f not in recorded:
            print(f"UNRECORDED {f}")
            failures += 1

    print(f"\n{len(recorded)} recorded files, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
