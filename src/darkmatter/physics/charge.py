"""Sequence charge statistics for intrinsically disordered regions.

Implementations are independent, vectorized (NumPy) versions of the
definitions in:

[1] Das, R.K. & Pappu, R.V. (2013) PNAS 110(33):13392-13397.
    doi:10.1073/pnas.1304749110  -- definitional source for kappa, delta,
    delta_max, SCD and the five diagram-of-states classes.
[2] Sawle, L. & Ghosh, K. (2015) J. Chem. Phys. 143:085101.
    doi:10.1063/1.4929361 -- SCD <-> chain compaction relation.

Charge convention: D/E carry q = -1; K/R carry q = +1; H is treated as
neutral at pH ~7 (standard Das-Pappu convention). All functions accept an
arbitrary upper-case amino-acid string; any character outside the
20-letter alphabet raises ValueError (fail closed, no silent parsing).

Every returned value is a plain float; sequences of length < 1 are
rejected. kappa() follows the exact Das-Pappu/localCIDER construction:
delta is averaged over sliding windows of 5 and 6 residues, and delta_max
is taken as the maximum delta over maximally charge-segregated benchmark
sequences built from the same residue composition. Parity with the
reference implementation (localcider) is enforced in tests.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "charges",
    "ncpr",
    "fcr",
    "f_plus",
    "f_minus",
    "sigma",
    "delta",
    "delta_max",
    "kappa",
    "scd",
    "charge_statistics",
]

POSITIVE = frozenset("KR")
NEGATIVE = frozenset("DE")
VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def charges(sequence: str) -> np.ndarray:
    """Return per-residue charges q_i in {-1, 0, +1} as float64."""
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("sequence must be a non-empty amino-acid string")
    if not set(sequence) <= VALID_AA:
        bad = sorted(set(sequence) - VALID_AA)
        raise ValueError(f"non-standard residue(s) {bad!r} in sequence")
    q = np.zeros(len(sequence), dtype=np.float64)
    for c in POSITIVE:
        q[np.frombuffer(sequence.encode(), dtype=np.uint8) == ord(c)] = 1.0
    for c in NEGATIVE:
        q[np.frombuffer(sequence.encode(), dtype=np.uint8) == ord(c)] = -1.0
    return q


def ncpr(sequence: str) -> float:
    """Net charge per residue, NCPR = (1/N) * sum(q_i)."""
    return float(charges(sequence).mean())


def f_plus(sequence: str) -> float:
    """Fraction of positively charged residues, f+ = N_+/N."""
    return float((charges(sequence) > 0).mean())


def f_minus(sequence: str) -> float:
    """Fraction of negatively charged residues, f- = N_-/N."""
    return float((charges(sequence) < 0).mean())


def fcr(sequence: str) -> float:
    """Fraction of charged residues, FCR = f+ + f-."""
    return float((charges(sequence) != 0).mean())


def sigma(sequence: str) -> float:
    """Charge asymmetry sigma = (f+ - f-)^2 / (f+ + f-); 0 when uncharged."""
    fp, fm = f_plus(sequence), f_minus(sequence)
    if fp + fm == 0.0:
        return 0.0
    return (fp - fm) ** 2 / (fp + fm)


def _delta_form(seq: np.ndarray, bloblen: int) -> float:
    """delta for one sliding-window size, per Das-Pappu Eq. (vectorized)."""
    n = seq.size
    if bloblen > n:
        return 0.0
    windows = np.lib.stride_tricks.sliding_window_view(seq, bloblen)
    bpos = (windows > 0).sum(axis=1)
    bneg = (windows < 0).sum(axis=1)
    bfcr = (bpos + bneg) / bloblen
    bncpr = (bpos - bneg) / bloblen
    bsig = np.where(bfcr == 0.0, 0.0, bncpr ** 2 / np.where(bfcr == 0.0, 1.0, bfcr))
    # sequence-level sigma from the original array: overlapping windows
    # multiply-count residues, so window-summed counts are invalid here
    seq_sigma = _seq_sigma_from_counts((seq > 0).sum(), (seq < 0).sum(), n)
    n_windows = n - bloblen + 1
    return float((((seq_sigma - bsig) ** 2) / n_windows).sum())


def _seq_sigma_from_counts(npos: float, nneg: float, n: int) -> float:
    if npos + nneg == 0:
        return 0.0
    fp, fm = npos / n, nneg / n
    return (fp - fm) ** 2 / (fp + fm)


def delta(sequence: str) -> float:
    """Average delta over sliding-window sizes 5 and 6 [1]."""
    seq = charges(sequence)
    return (_delta_form(seq, 5) + _delta_form(seq, 6)) / 2.0


def _benchmark_delta_max(npos: int, nneg: int, nneut: int) -> float:
    """Max delta over maximally-segregated benchmark sequences.

    Replicates the four construction cases of the reference algorithm:
    only-positive/only-negative, no neutrals, >= 18 neutrals (start/mid/end
    sweep), and the full general permutation search.
    """
    dmax = 0.0
    seqs: list[str] = []
    if nneg == 0 or npos == 0:
        ncharged = max(npos, nneg)
        charge_char = "+" if npos > 0 else "-"
        if nneut >= ncharged:
            for position in range(nneut - ncharged + 1):
                seqs.append("0" * position + charge_char * ncharged
                            + "0" * (nneut - position - ncharged))
        else:
            for position in range(ncharged - nneut + 1):
                seqs.append(charge_char * position + "0" * nneut
                            + charge_char * (ncharged - position))
    elif nneut == 0:
        pos_block, neg_block = "+" * npos, "-" * nneg
        if npos > nneg:
            for position in range(nneg + 1):
                seqs.append("+" * position + neg_block + "+" * (npos - position))
        else:
            for position in range(npos + 1):
                seqs.append("-" * position + pos_block + "-" * (nneg - position))
    elif nneut >= 18:
        pos_block, neg_block = "+" * npos, "-" * nneg
        for start in range(7):
            for end in range(7):
                if start + end > nneut:
                    continue
                seqs.append("0" * start + pos_block + "0" * (nneut - start - end)
                            + neg_block + "0" * end)
    else:
        pos_block, neg_block = "+" * npos, "-" * nneg
        for mid in range(nneut + 1):
            for start in range(nneut - mid + 1):
                seqs.append("0" * start + pos_block + "0" * mid + neg_block
                            + "0" * (nneut - start - mid))
    for s in seqs:
        arr = np.zeros(len(s), dtype=np.float64)
        a = np.frombuffer(s.encode(), dtype=np.uint8)
        arr[a == ord("+")] = 1.0
        arr[a == ord("-")] = -1.0
        d = (_delta_form(arr, 5) + _delta_form(arr, 6)) / 2.0
        dmax = max(dmax, d)
    return dmax


def delta_max(sequence: str) -> float:
    """Maximum possible delta for the same residue composition [1]."""
    q = charges(sequence)
    npos = int((q > 0).sum())
    nneg = int((q < 0).sum())
    nneut = q.size - npos - nneg
    if npos + nneg == 0:
        return 0.0
    return _benchmark_delta_max(npos, nneg, nneut)


def kappa(sequence: str) -> float:
    """Das-Pappu charge patterning parameter kappa in [0, 1] [1].

    kappa = delta / delta_max. Returns 0.0 for sequences with no charged
    residues. Small overflow of delta over delta_max (kappa in (1, 1.1),
    a documented reference-implementation heuristic) is clamped to 1.0;
    larger excursions raise, so numerical bugs cannot pass silently.
    """
    q = charges(sequence)
    if not (( q > 0) | (q < 0)).any():
        return 0.0
    k = delta(sequence) / delta_max(sequence)
    if k > 1.0 and k < 1.1:
        return 1.0
    if k > 1.1:
        raise FloatingPointError(f"kappa exceeded 1.1 ({k}); delta_max construction bug")
    return float(k)


def scd(sequence: str) -> float:
    """Sequence Charge Decoration [1, 2]: (1/N) * sum_{i<j} q_i q_j sqrt(j-i)."""
    q = charges(sequence)
    n = q.size
    if n < 2:
        return 0.0
    ii, jj = np.triu_indices(n, k=1)
    dist = (jj - ii) ** 0.5
    return float((q[ii] * q[jj] * dist).sum() / n)


def charge_statistics(sequence: str) -> dict:
    """Convenience bundle of all charge statistics."""
    return {
        "length": len(sequence),
        "ncpr": ncpr(sequence),
        "fcr": fcr(sequence),
        "f_plus": f_plus(sequence),
        "f_minus": f_minus(sequence),
        "sigma": sigma(sequence),
        "kappa": kappa(sequence),
        "scd": scd(sequence),
        "delta": delta(sequence),
        "delta_max": delta_max(sequence),
    }
