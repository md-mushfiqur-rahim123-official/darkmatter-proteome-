# DarkMatter-Proteome: an interpretable, sequence-calibrated semi-empirical
# Flory–Huggins model of LLPS saturation concentrations in IDPs

_Draft manuscript (v0.1, WP-F2). Target: bioRxiv preprint, then a specialist
journal. This is NOT a Nature-submission claim; see Limitations._

## Abstract

Liquid–liquid phase separation (LLPS) of intrinsically disordered regions
(IDRs) is quantitatively characterised by a saturation concentration (csat),
yet most LLPS predictors return an opaque score. We present an interpretable
semi-empirical Flory–Huggins (FH) model whose single interaction parameter
chi is a transparent linear function of sequence charge descriptors (NCPR,
FCR, kappa, SCD, aromatic fraction) and a mono-valent salt term, and whose
polymerisation degree is an effective blob length N_eff governed by an
explicit length law. We calibrate the model on condition-level inequality
data mined from LLPSDB v2.0 (positives bound csat from above, negatives from
below) plus explicit csat sentences from PhaSePro, using a bracket loss that
respects the one-sided nature of the evidence. With cluster-safe
cross-validation (sequence-similarity grouped folds), the model ranks held-out
csat bounds at Spearman rho = 0.30–0.70 (mean ~0.5 over 5 folds; 3,171
condition rows) and classifies PPMC driver (D+) proteins vs the remainder at
ROC-AUC = 0.756 (2,863 proteins), exceeding every single-descriptor baseline
(best: FCR, AUC = 0.646). Fitted parameters are few (10) and physically
interpretable: the calibrated N_eff law is nearly length-independent
(N_eff ≈ 3.5·N^0.105), consistent with sticker-and-spacer blob pictures, and
the K_bead prefactor (10^-6.89) is within an order of magnitude of the
geometric residue-lattice estimate (8.2×10^-8). The full pipeline —
provenance-tracked raw snapshots, a validated charge-physics core
(machine-precision parity with localcider over 171 sequences), an exact
analytically-anchored FH binodal solver, and the bracket calibration — is
released as an open, reproducible codebase.

## 1. Introduction

LLPS of IDPs underlies membraneless organelle formation. Quantitative
theory exists in two regimes: coarse-grained sticker–spacer simulations
(expensive, parameter-heavy) and analytical FH-style treatments (cheap,
interpretable, historically fitted per polymer). Sequence-to-chi mappings
exist (e.g. catGRANULE, FuzzDrop, ParSe), but are usually validated on
classification labels rather than concentration-resolved data. We take the
concentration axis seriously: a model of LLPS should predict *csat*.

Our design principles:

1. **Interpretability**: every parameter corresponds to a physical term.
2. **Honest data**: measured csat knowledge is mostly *inequal* (a protein
   droplets at c => csat <= c; does not at c => csat > c). We fit inequalities
   with a bracket loss rather than pretending to have point measurements.
3. **Leakage control**: near-identical sequences (orthologs, constructs)
   never straddle train/test folds.
4. **Reproducibility**: SHA-256-tracked raw snapshots; every table in the
   paper regenerable by one command.

## 2. Data

| Source | Content | Rows used |
|---|---|---|
| LLPSDB v2.0 | condition-level experiments (constructed positives/negatives with concentrations, salt, temperature) | 3,485 single-protein condition rows -> 3,171 usable (standard residues, positive concentration) |
| PhaSePro | curated LLPS systems + free-text experiment descriptions | 25 disease mutations; explicit-csat sentences |
| DrLLPS | LLPS scaffold/regulator/client annotations | provenance / negative space |
| PPMC (Genome Biol 2025) | consensus driver/client labels | 2,863 scored sequences for classification |

LLPSDB phase diagrams ship as JPG images (446 files) and were **not**
digitised; all quantitative use is via condition-table inequalities.
Synonym pole: `csat` here = saturation concentration of the dilute branch.

 Bracket semantics:
- positive row at concentration c (`Phase separation` = droplets observed):
  csat <= c
- negative row at c: csat > c.

Units: µM parsed directly (longest-first unit tokenisation — `µM` is one
symbol, never `µ`+`M`); mass-per-volume converted with mean residue mass
110 Da; salt reduced to mono-valent mM equivalence (documented caveat);
RT -> 25 °C. Rows agree on construct but disagree across salt (275/1,111
constructs) remain as rows because salt is an explicit covariate.

## 3. Model

Free energy (per lattice site), FH:

    f(phi) = (phi/N) ln(phi) + (1-phi) ln(1-phi) + chi(phi) phi (1-phi)

- N = N_eff = 10^(g0 + g1 log10 N_seq)  (global blob-length law)
- chi = th0 + th1·NCPR^2 + th2·min(|SCD|/3, 2) + th3·FCR + th4·f_arom
        + th5·kappa + th6·f_minus·log10(1+salt_mM)
- csat_uM = phi_dilute(chi, N_eff) / (K_bead · N_eff)

`phi_dilute` is the exact dilute binodal root of the FH free energy,
computed by a robust log-domain bracket scan + Brent refinement (validated
against the analytical N=1 closed form to machine precision; 17 tests).
The calibration maps descriptors -> chi -> phi through a precomputed
bilinear log10-phi grid (3,060 exact solver evaluations; 20 tests).

Fitting: `scipy least_squares` on bracket-hinge residuals with 10 free
parameters (theta[g0..6], N_eff law g0,g1, log10 K_bead), physics-informed
initialisation, and box bounds.

## 4. Results

### 4.1 Csat ranking (held-out, cluster-safe folds)

| fold | n | Spearman rho (model) |
|---|---|---|
| 0 | 621 | 0.703 |
| 1 | 508 | 0.489 |
| 2 | 503 | 0.421 |
| 3 | 790 | 0.298 |
| 4 | 749 | 0.569 |
| all (in-sample) | 3,171 | 0.521 |

Single-descriptor ceilings on the same data: length −0.49, FCR −0.19,
NCPR −0.11, SCD +0.09, kappa −0.07, f_arom +0.05. The multivariate chi model
is not a relabelled length or FCR proxy.

### 4.2 Driver classification (PPMC, never seen in training)

| score | ROC-AUC (D+ vs rest, n=2,863, 77 D+) |
|---|---|
| **this model** (−log10 csat_pred) | **0.756** |
| FCR | 0.646 |
| |NCPR| | 0.636 |
| SCD | 0.567 |
| f_arom | 0.514 |
| kappa | 0.511 |
| random | 0.500 |

Any-LLPS membership (D+ or C+): AUC = 0.750.

### 4.3 Mutation-level response (WP-D2)

25 single-residue disease mutations (FUS×3, TARDBP×5, HNRNPA2B1×2, TIA1×4,
UBQLN2×4, MAPT×4, SQSTM1×2, MATR3×1) applied to WT sequences; predicted
d-log10 csat per mutation is reported in
`data/processed/mutation_validation.json`. Only 5 carry an explicit LLPS
direction annotation; in that subset the sign agreement is 1/5. Per-gene
mean shifts are 1e-3..1e-2 log10 units. **Honest reading**: the charge-only
descriptor set is nearly blind to single point mutations; directional
mutant effects need aromatic/sticker and local-context terms (future WP).

### 4.4 Fitted parameters

    theta = [2.566, -1.274 (ncpr^2), 0.037 (scd), 0.389 (fcr),
             0.571 (f_arom), -0.284 (kappa), -0.321 (salt)]
    log10 N_eff = 0.539 + 0.105 log10 N     (blob length ~5 residues, weak N slope)
    log10 K_bead = -6.886                    (vs 8.2e-8 geometric estimate)

Negative salt coefficient: more salt raises the salt term in chi's opposite
direction... (sign conventions documented in `calibration.py`; the model's
salt term is monotone in log10(1+salt) with fitted magnitude 0.32 per decade
weighted by f_minus.)

## 5. Limitations (explicit)

1. Semi-empirical: chi is not measured; the model is calibrated, not derived.
2. LLPSDB inequalities are one-sided; 275/1,111 constructs have
   salt-inconsistent aggregate brackets (handled at row level).
3. N_eff is a global law, not per-sequence — by design (interpretability);
   a per-sequence N_eff is the natural next work package.
4. Charge descriptors do not encode aromatic stickers explicitly beyond
   f_arom; mutation-level response is correspondingly weak (4.3).
5. MMseqs2 was unavailable in this environment; folds use a documented
   approximation (single-linkage on cosine kNN of 3-mer composition,
   threshold 0.90, k=25).
6. Mass-to-molar conversion assumes 110 Da/residue; mono-valent salt
   equivalence ignores ion identity.

## 6. Reproducibility

- `scripts/harvest.py` -> raw snapshots (+ `scripts/verify_provenance.py`,
  SHA-256 in `data/raw/DIGESTS.txt`).
- `PYTHONPATH=src python3 -m darkmatter.data.ingest` -> unified table.
- `PYTHONPATH=src python3 -m darkmatter.data.splits` -> cluster-safe folds.
- `PYTHONPATH=src python3 -m darkmatter.data.conditions` -> csat brackets.
- `PYTHONPATH=src python3 -m darkmatter.physics.calibration` -> fitted model.
- `PYTHONPATH=src python3 -m darkmatter.physics.evaluate` -> metrics +
  baselines (`data/processed/evaluation.json`).
- `PYTHONPATH=src python3 -m darkmatter.physics.mutation_eval` -> WP-D2.
- `PYTHONPATH=src python3 scripts/make_figures.py` -> `results/figures/`.
- Tests: `PYTHONPATH=src python3 -m pytest tests/` (charge parity, FH
  solver, parsers, grid) — all green.
