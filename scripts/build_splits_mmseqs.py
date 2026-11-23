"""WP-G: rebuild leakage-controlled folds from real MMseqs2 linclust clusters.

Requires data/processed/unique_seqs.fasta + mmseqs_clusters_cluster.tsv
(produced by:
  tools/mmseqs/bin/mmseqs easy-linclust data/processed/unique_seqs.fasta \
      data/processed/mmseqs_clusters data/processed/mmseqs_tmp \
      --min-seq-id 0.3 -c 0.85 --cov-mode 5 --threads 4 )
cov-mode 5 + coverage 0.85 => the SHORTER sequence must be >=85% covered, so
fragments of the same protein can never straddle two folds.

Run: PYTHONPATH=src python3 scripts/build_splits_mmseqs.py
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from darkmatter.data.splits import assign_folds  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC = os.path.join(ROOT, "data", "processed")


def main():
    old = json.load(open(os.path.join(PROC, "splits_knn_approx.json")))
    recs = old["records"]
    rep_of = {}
    for line in open(os.path.join(PROC, "mmseqs_clusters_cluster.tsv")):
        rep, mem = line.split()
        rep_of[mem] = rep
    assert len(rep_of) == len(recs), (len(rep_of), len(recs))
    reps = sorted(set(rep_of.values()))
    cid = {rep: i for i, rep in enumerate(reps)}
    folds = assign_folds([cid[rep_of[r["id"]]] for r in recs],
                         n_folds=5, seed=20250101)
    new = {
        "method": ("MMseqs2 easy-linclust v20-ae60e (min-seq-id 0.3, "
                   "coverage 0.85, cov-mode 5 = shorter-seq coverage; "
                   "fragment-leakage-safe)"),
        "binary_sha256": "f6ab59defde7d65ea7d55dd33aed167ada87cdae8b6b85fd4ba2dc6f00aa7775",
        "similarity_threshold": 0.3, "coverage": 0.85, "cov_mode": 5,
        "n_folds": 5, "seed": 20250101,
        "replaces": {"method": "3-mer composition kNN approximation",
                     "note": "archived at splits_knn_approx.json"},
        "records": [{**r, "fold": f} for r, f in zip(recs, folds)],
    }
    sizes = Counter(rep_of[r["id"]] for r in recs)
    print("fold sizes:", dict(sorted(Counter(folds).items())),
          "| n_clusters:", len(sizes), "largest:", max(sizes.values()))
    json.dump(new, open(os.path.join(PROC, "splits_mmseqs.json"), "w"))
    idx = {k: v for k, v in new.items() if k != "records"}
    idx["records"] = [{k: v for k, v in r.items() if k != "sequence"}
                      for r in new["records"]]
    json.dump(idx, open(os.path.join(PROC, "splits_index.json"), "w"))
    print("wrote splits_mmseqs.json + splits_index.json")


if __name__ == "__main__":
    main()
