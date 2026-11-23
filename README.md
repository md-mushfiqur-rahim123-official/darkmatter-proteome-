# DarkMatter-Proteome

Interpretable, sequence-calibrated semi-empirical computational engine for
**liquid–liquid phase separation (LLPS) of intrinsically disordered
proteins (IDPs)** — built on charge patterning (κ, SCD of Das–Pappu 2013 /
Sawle–Ghosh 2015), sticker chemistry, and mean-field Flory–Huggins
coexistence physics, calibrated and evaluated **only** against
experimentally curated public ground truth.

**Honest positioning statement:** this is *not* a "first-principles from
scratch" predictor and makes no such claim — sequence-to-χ calibration is
semi-empirical by construction; the *engine* (free-energy functional,
common-tangent coexistence solver, charge-statistics kernels) is exact
analytical mean-field physics, validated to machine precision. The research
goal is **interpretable Csat ranking + calibratable phase boundaries**, not
a black-box SOTA claim against DeePhase/FuzDrop/catGRANULE (those are
baselines to compare against, not competitors to disparage).

## Status (WP-B1/B3/D1/B2/E1/E2/D2/F1/F2/G complete)

| Work package | Status |
|---|---|
| WP-B1 charge-statistics core (κ, SCD, NCPR, FCR, σ, δ, δ_max) | ✅ validated to machine precision vs `localcider` (171/171 sequences, max\|Δκ\| ≈ 1.2e-14) |
| WP-B2 χ(seq)/N_eff calibration (LLPSDB condition brackets) | ✅ 10-param fit, 5-fold MMseqs2 cluster-safe CV ρ=0.34–0.58 (mean 0.444) |
| WP-E1 evaluation harness | ✅ PPMC D+ ROC-AUC **0.756** vs baselines ≤0.646 |
| WP-E2 baselines | ✅ FCR 0.646, \|NCPR\| 0.636, SCD 0.567, f_arom 0.514, κ 0.511 |
| WP-D2 mutation-level validation | ✅ 25 mutations scored; honest negative (charge-only blind to point mutants) |
| WP-F1 figures | ✅ `results/figures/` |
| WP-F2 manuscript | ✅ draft: `MANUSCRIPT.md` |
| WP-G MMseqs2 | ✅ **resolved**: v20-ae60e static binary (SHA-256-tracked); folds from real MMseqs2 linclust (0.3 id, cov-mode 5) — 7,954 clusters |
| WP-B3 Flory–Huggins binodal/spinodal solver | ✅ validated (analytical χ_c/φ_c, common-tangent verified, N=1 closed-form match) |
| 4-source data ingest + leakage-controlled splits | ✅ 13,221 unique sequences; MMseqs2 linclust: 7,954 clusters, 5 folds |
| WP-D1 pathological mutation benchmark | ✅ 25 applied mutations (FUS×3, TDP-43×5 + 17 more) |

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas scikit-learn xlrd pytest localcider

# rebuild all processed artefacts from raw snapshots
PYTHONPATH=src python3 -m darkmatter.data.ingest      # unified_raw_sources.json
PYTHONPATH=src python3 -m darkmatter.data.splits      # splits.json  (5-fold)
PYTHONPATH=src python3 -m darkmatter.data.mutations   # mutation_benchmark.json

# physics demo
PYTHONPATH=src python3 -c "from darkmatter.physics.flory_huggins import csat; print(csat(1.5, 25.0))"

# provenance: verify every raw snapshot against its SHA-256 ledger
python3 scripts/verify_provenance.py

# tests
PYTHONPATH=src python3 -m pytest tests/ -q
```

## Data

Raw snapshots are **immutable**; every byte is verified against
`data/raw/DIGESTS.txt` (SHA-256 ledger, enforced by
`scripts/verify_provenance.py`). Full source URLs, timestamps and citation
notes live in [`data/PROVENANCE.md`](data/PROVENANCE.md).

| Source | Role | Size after ingest |
|---|---|---|
| PhaSePro (2020) | curated LLPS drivers + disease-mutation annotations | 121 sequences |
| DrLLPS (2020) | Scaffold/Regulator/Client labels | 9,281 entries |
| PPMC (Genome Biol 2025) | consensus D+/D−/C+/C− + non-LLPS negatives | 2,876 entries |
| LLPSDB v2.0 | condition-level tested positives/negatives + **quantitative phase diagrams** (χ-calibration set) | 1,956 unique sequences |

### Leakage-controlled splits

`MMseqs2` is not available in the current execution sandbox, so
`darkmatter.data.splits` ships a deterministic, documented **approximation**:
single-linkage clustering over a cosine k-NN graph of L2-normalised 3-mer
composition vectors (threshold 0.90, k=25). On this corpus it yields 11,304
clusters over 13,221 unique standard-residue sequences with a largest
cluster of 12 — ortholog families merge; compositionally similar,
non-homologous IDPs stay separate. **Before publication, the clustering
should be swapped to MMseqs2 `easy-cluster` at 30% identity** (tracked in
PROVENANCE "known gaps"); `assign_folds` and the `splits.json` schema are
clustering-agnostic.

## Layout

```
src/darkmatter/physics/charge.py         κ/SCD/NCPR/FCR/σ/δ/δ_max (localcider-parity)
src/darkmatter/physics/flory_huggins.py  FH free energy, spinodal, χ_c, binodal, csat,
                                         χ ← csat inverse calibration entry point
src/darkmatter/data/ingest.py            4-source unified ingestion
src/darkmatter/data/splits.py            homology-clustered 5-fold splits
src/darkmatter/data/mutations.py         PhaSePro-disease mutation benchmark
src/darkmatter/data/conditions.py        LLPSDB condition -> csat bracket pipeline
src/darkmatter/physics/calibration.py    chi/N_eff bracket-loss fit + standalone runner
src/darkmatter/physics/evaluate.py       eval harness + baseline ablations
src/darkmatter/physics/mutation_eval.py  mutation-level csat response (WP-D2)
scripts/make_figures.py                  publication figures -> results/figures/
scripts/harvest.py                       idempotent provenance-controlled re-download
scripts/verify_provenance.py             SHA-256 verification of raw snapshots
tests/                                   37 pytest (charge parity + FH analytics + parsers + grid)
```

## Design decisions worth knowing

1. **N_eff, not N.** With the textbook entropy term φ/N·ln φ, chain-length
   N≈600 pushes the dilute binodal to φ ~ e^(−hundreds) — physically
   correct for long polymers but useless for IDP Csat (μM range). IDP
   condensate modelling therefore uses an EFFECTIVE sticker/blob length
   N_eff ≪ N, calibrated per sequence — that calibration is exactly WP-B2.
2. **Charge convention:** K/R = +1, D/E = −1, H = 0 (matches Das–Pappu);
   the sequence-level σ must be computed from the raw charge array, never
   from overlapping-window counts (a real bug caught by parity testing).
3. **No p-value theatre:** descriptor-shift outputs (ΔSCD, Δκ) are reported
   with their deterministic values alongside the annotation direction from
   PhaSePro; statistical claims wait for the calibrated model. Note that
   charge-neutral mutations (FUS P525L, TDP-43 M337V) have ΔSCD = 0 by
   construction — they will need non-charge features (aromatic sticker,
   disorder propensity), not inflated claims.
4. **Strict residue alphabet:** sequences with non-standard residues are
   excluded from splits (records keep provenance for later re-inclusion).
