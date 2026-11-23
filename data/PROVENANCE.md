# Data Provenance Ledger

Every raw dataset used by DarkMatter-Proteome is recorded here with its
source URL, fetch timestamp (UTC), SHA-256 digest, and licensing notes.
Per the project truth law: no dataset enters the pipeline without an entry
on this ledger. `scripts/verify_provenance.py` re-verifies digests.

| File | Source | Fetched (UTC) | SHA-256 | Notes |
|---|---|---|---|---|
| `data/raw/phasepro_full.json` | `https://phasepro.elte.hu/download_full.json` (PhaSePro, Mészáros et al., Nucleic Acids Res. 2020, 48:D360–D367, doi:10.1093/nar/gkz848) | 2026-11-14 (this session; exact timestamp + digest below once verified by script) | *(see `data/raw/DIGESTS.txt`)* | 121 curated LLPS driver entries (59 human); full sequences, driver-region boundaries, PMIDs, quantitative Csat text, disease-mutation annotations with direction. |
| `data/raw/drllps_LLPS.txt` | `http://llps.biocuckoo.cn/download/LLPS.txt` (DrLLPS, Ning et al., Nucleic Acids Res. 2020, 48:D288–D295, doi:10.1093/nar/gkz1027) | same session | *(see `data/raw/DIGESTS.txt`)* | 9,281 curated entries: 150 Scaffold / 987 Regulator / 8,144 Client with sequences; human subset 3,633. |
| PPMC confident datasets | `https://github.com/PPMC-lab/llps-datasets` (Pintado-Grima et al., Genome Biology 26:198, 2025, doi:10.1186/s13059-025-03668-6; Zenodo doi:10.5281/zenodo.15118996) | same session, cloned as `github__PPMC-lab__llps-datasets`, snapshot copied via `scripts/harvest.py --include-ppmc` | *(see `data/raw/DIGESTS.txt`, `ppmc_datasets.tsv`)* | 2,876 proteins with unambiguous driver / client / negative labels — our primary benchmark negatives (IDR-aware, avoids the folded-vs-disordered confound). |
| `data/raw/Phase_separation_*.zip` | `http://bio-comp.org.cn/llpsdbv2/download/` (LLPSDB v2.0, Wang et al., doi:10.1093/database/miac?{verify on citation}; download dir listing) | same session, digests in `DIGESTS.txt` | *(see DIGESTS.txt)* | 4,111 + 160 condition-level positive LLPS entries (protein sequences + experimental conditions, both study arms). |
| `data/raw/No_phase_separation_*.zip` | same as above | same session | *(see DIGESTS.txt)* | 1,894 + 65 condition-level **tested non-phase-separating** entries — sequence-matched true negatives. |
| `data/raw/Phase_diagram_*.zip` | same as above | same session | *(see DIGESTS.txt)* | 444 + 4 entries with **quantitative phase diagrams** — the χ calibration set. |

All nine snapshot files are digest-verified by `scripts/verify_provenance.py`
(exits 0 = all digests match).

## Known gaps / next collection steps

- **LLPSDB v2.0**: all six zips harvested and digest-verified. LLPS.xls has
  TWO different schemas (Unambiguous vs Ambiguous) — see
  `src/darkmatter/data/ingest.py::load_llpsdb` for the handling; Ambiguous
  zips store sequences as ';'-joined FASTA lines and Lack a Sequence column
  in LLPS.xls (joined from protein.xls via protID↔PID).
- **MMseqs2**: not available in the current sandbox; splits use the
  documented 3-mer-composition kNN approximation
  (`darkmatter.data.splits`). Swap to MMseqs2 `easy-cluster` (30% identity)
  before publication; `splits.json` schema is clustering-agnostic.
- **DisProt**: not yet harvested; used later only as an IDR mask/filter.
- LLPSDB citation: exact NAR-Database article to be confirmed when writing the
  manuscript bibliography (site lists the paper; DOI to be verified — no
  invented DOIs on this ledger beyond ones seen in source pages).
- Snapshot copies of database home pages saved alongside raw data for
  reproducibility (`*_download.html`).

## Rules

1. Raw snapshots are immutable; processing happens in `data/processed/`.
2. Any sequence used in training/evaluation must be traceable to a row here.
3. Re-verification: `python scripts/verify_provenance.py` (recomputes SHA-256
   against this ledger; exits non-zero on any drift).

## WP-B2 quantitative path (2025-11 session)

- LLPSDB "phase diagram" zips ship 446 JPG images + condition tables. JPGs
  are NOT digitised; the quantitative path is the CONDITION-LEVEL inequality
  data (see `src/darkmatter/data/conditions.py`).
- Bracket semantics: single-protein positive row at c => csat <= c;
  single-protein negative row at c => csat > c. Output:
  `data/processed/llpsdb_conditions.json`
  (3,485 condition rows -> 1,111 constructs -> 1,039 csat-bounded;
  275 constructs aggregate-inconsistent across salt — kept, salt is an
  explicit covariate in the chi model).
- Units: longest-first unit tokenisation (µM is a single unit token —
  regression-tested); mass/volume -> µM via 110 Da/residue
  (AA_MASS_DA, documented approximation); salt -> mono-valent mM
  equivalence (documented caveat); RT -> 25 °C.
- PhaSePro free-text explicit csat sentences parsed by phrase requirement
  ("saturation concentration"/"csat") + asserted-number check; currently
  yields 1 high-precision point (FUS).
- Calibration output: `data/processed/calibration_result.json`
  (10 params; cluster-safe 5-fold CV; see MANUSCRIPT.md for numbers).
- Evaluation output: `data/processed/evaluation.json`;
  mutation validation: `data/processed/mutation_validation.json`.
- WP-I uncertainty addendum: bootstrap CIs (1000 resamples, seed 20250101)
  appended to `data/processed/evaluation.json`
  (csat ranking within-fold resampling; DrLLPS scaffold AUC 0.590
  [0.548, 0.629] on 9,278 proteins — third external label set).
