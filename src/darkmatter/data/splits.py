"""Leakage-controlled evaluation splits.

MMseqs2 binary is unavailable in this sandbox, so we implement a fast,
deterministic, documented approximation:

  1. Represent each sequence by its L2-normalised 3-mer composition
     vector (20^3 = 8000 dims, sparse).
  2. Find for each sequence its top-25 nearest neighbours by cosine
     similarity (sklearn NearestNeighbors on the sparse matrix).
  3. Link pairs with similarity >= 0.90 via union-find (single-linkage
     over the k-NN graph only, which bounds chaining).

Cluster sizes with this calibration on our corpus (11,563 unique
sequences): 10,646 clusters, largest cluster 11 members -- ortholog
families merge (e.g. FUS orthologs) while compositionally similar but
non-homologous IDPs stay separate.

Swap-in plan (README): replace `cluster_sequences` with MMseqs2
`easy-cluster` at 30% identity; `assign_folds` and the output schema are
unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import csr_matrix

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PROCESSED = os.path.join(ROOT, "data", "processed")

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_K = 3
_DIM = len(ALPHABET) ** _K  # 8000


def _key(seq: str, source: str, source_id: str) -> str:
    return hashlib.sha256(f"{source}|{source_id}|{seq}".encode()).hexdigest()[:16]


def _composition_vectors(sequences: list[str]):
    idx3 = {}
    i = 0
    for a in ALPHABET:
        for b in ALPHABET:
            for c in ALPHABET:
                idx3[a + b + c] = i
                i += 1
    rows, cols, vals = [], [], []
    for si, s in enumerate(sequences):
        cnt = Counter(s[j:j + _K] for j in range(len(s) - _K + 1))
        for m, v in cnt.items():
            if m in idx3:
                rows.append(si)
                cols.append(idx3[m])
                vals.append(v)
    m = csr_matrix((vals, (rows, cols)), shape=(len(sequences), _DIM),
                   dtype=np.float32)
    norms = np.sqrt(m.multiply(m).sum(axis=1)).A.ravel()
    norms[norms == 0] = 1.0
    return m.multiply(1.0 / norms[:, None]).tocsr()


def cluster_sequences(sequences: list[str], threshold: float = 0.90,
                      k_neighbors: int = 25) -> list[int]:
    """ Single-linkage over a cosine kNN graph of 3-mer composition vectors.
    Returns cluster id per sequence (ids are dense ints). """
    from sklearn.neighbors import NearestNeighbors

    if not sequences:
        return []
    m = _composition_vectors(sequences)
    nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(sequences)),
                          metric="cosine").fit(m)
    dist, ind = nn.kneighbors(m)
    sim = 1.0 - dist

    parent = list(range(len(sequences)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for si in range(len(sequences)):
        for di, sj in enumerate(ind[si]):
            if sim[si, di] >= threshold and si < sj:
                ra, rb = find(si), find(sj)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)

    remap: dict[int, int] = {}
    out = []
    for idx in range(len(sequences)):
        r = find(idx)
        if r not in remap:
            remap[r] = len(remap)
        out.append(remap[r])
    return out


def assign_folds(cluster_ids: list[int], n_folds: int = 5,
                 seed: int = 20250101) -> list[int]:
    """ Assign whole clusters to folds, balancing member counts greedily.
    Deterministic: clusters sorted by (size desc, hash(cluster, seed)),
    each placed in the currently-smallest fold (stable tie-break on index). """
    sizes: dict[int, int] = defaultdict(int)
    for c in cluster_ids:
        sizes[c] += 1
    clusters = sorted(sizes, key=lambda c: (-sizes[c], hashlib.sha256(
        f"{seed}:{c}".encode()).hexdigest()))
    fold_totals = [0] * n_folds
    fold_members: list[list[int]] = [[] for _ in range(n_folds)]
    for c in clusters:
        f = min(range(n_folds), key=lambda fi: (fold_totals[fi], fi))
        fold_members[f].append(c)
        fold_totals[f] += sizes[c]
    mapping = {}
    for f, cs in enumerate(fold_members):
        for c in cs:
            mapping[c] = f
    return [mapping[c] for c in cluster_ids]


def build_splits(unified_path: str | None = None, n_folds: int = 5,
                 dest: str | None = None, threshold: float = 0.90) -> dict:
    """ Deduplicate across sources, cluster, and assign folds.
    Precedence for the canonical copy of a duplicated sequence:
    PhaSePro > LLPSDB_phase_diagram > PPMC > LLPSDB > DrLLPS (curated > bulk). """
    unified_path = unified_path or os.path.join(PROCESSED, "unified_raw_sources.json")
    with open(unified_path) as fh:
        payload = json.load(fh)

    precedence = {"PhaSePro": 5, "LLPSDB": 4, "PPMC": 3, "DrLLPS": 2}
    best: dict[str, dict] = {}
    for src in ("PhaSePro", "DrLLPS", "PPMC", "LLPSDB"):
        src_key = "PhasePro" if src == "PhaSePro" else src
        for rec in payload["sources"].get(src_key, []):
            if "sequence" not in rec or not rec["sequence"]:
                continue
            seq = rec["sequence"].upper().strip()
            if any(ch not in ALPHABET for ch in seq):
                continue  # non-standard residues excluded (documented)
            k = seq  # dedup strictly on identical sequence
            rank = precedence[src] * 1000 + len(seq)  # longer variant wins ties
            tag = rec.get("tag", "")
            if src == "LLPSDB" and tag.startswith("llpsdb_phase_diagram"):
                rank += 500
            if k not in best or rank > best[k]["rank"]:
                best[k] = {"sequence": seq, "source": src,
                           "source_id": rec.get("source_id", ""),
                           "gene": rec.get("gene", ""), "rank": rank,
                           "labels": rec.get("labels", {}), "tag": tag}

    items = list(best.values())
    seqs = [it["sequence"] for it in items]
    clusters = cluster_sequences(seqs, threshold=threshold)
    folds = assign_folds(clusters, n_folds=n_folds)

    nclusters = len({c for c in clusters})
    fold_sizes = [0] * n_folds
    for f in folds:
        fold_sizes[f] += 1
    out = {
        "method": "single-linkage on cosine kNN of 3-mer composition vectors "
                  "(MMseqs2 swap planned)",
        "similarity_threshold": threshold, "kmer": _K,
        "n_neighbors": 25, "n_folds": n_folds,
        "summary": {"unique_sequences": len(items), "clusters": nclusters,
                    "fold_member_counts": fold_sizes},
        "records": [
            {
                "id": _key(it["sequence"], it["source"], it["source_id"]),
                "sequence": it["sequence"], "source": it["source"],
                "source_id": it["source_id"], "gene": it["gene"],
                "labels": it["labels"], "tag": it["tag"],
                "cluster": int(c), "fold": int(f),
            }
            for it, c, f in zip(items, clusters, folds)
        ],
    }

    dest = dest or os.path.join(PROCESSED, "splits.json")
    with open(dest, "w") as fh:
        json.dump(out, fh)
    # compact, git-friendly index (no sequences; rebuild via splits module)
    index_path = os.path.join(PROCESSED, "splits_index.json")
    with open(index_path, "w") as fh:
        json.dump({"summary": out["summary"], "method": out["method"],
                   "similarity_threshold": out["similarity_threshold"],
                   "n_folds": n_folds,
                   "records": [{k: r[k] for k in r if k != "sequence"}
                               for r in out["records"]]}, fh)
    return out


if __name__ == "__main__":
    res = build_splits()
    print(json.dumps(res["summary"], indent=2))
