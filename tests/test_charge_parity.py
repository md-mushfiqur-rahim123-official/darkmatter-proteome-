"""Parity + invariant tests for the charge-statistics core.

Reference implementation: localcider (Pappu lab). Every new release must
reproduce localcider's kappa/SCD to machine precision on real ground-truth
sequences (PhaSePro snapshot) and randomized sequences. Run with:
    PYTHONPATH=src python -m pytest tests/ -q
localcider is a test-only dependency; the shipped package does not use it.
"""
import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from darkmatter.physics.charge import (  # noqa: E402
    charge_statistics,
    fcr,
    kappa,
    ncpr,
    scd,
)

DATA = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "phasepro_full.json"
)

localcider = pytest.importorskip("localcider.sequenceParameters")


def _sequences():
    with open(DATA) as fh:
        data = json.load(fh)
    seqs = [v["sequence"].replace("\n", "") for v in data.values()]
    rng = random.Random(42)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    seqs += [
        "".join(rng.choice(alphabet) for _ in range(rng.randint(20, 300)))
        for _ in range(50)
    ]
    return seqs


def test_kappa_and_scd_match_localcider():
    from localcider.sequenceParameters import SequenceParameters

    for seq in _sequences():
        try:
            ref = SequenceParameters(seq)
            ref_kappa = ref.get_kappa()
            ref_scd = ref.get_SCD()
        except Exception:
            continue
        if ref_kappa == -1 or ref_kappa is None:
            continue
        assert abs(kappa(seq) - ref_kappa) < 1e-9, seq[:50]
        assert abs(scd(seq) - ref_scd) < 1e-9, seq[:50]


def test_scid_hand_computed_value():
    # q = [+1,-1]; SCD = (1/2) * 1*(-1) * sqrt(1) = -0.5
    assert scd("KD") == pytest.approx(-0.5)


def test_kappa_extremes_and_symmetry():
    segregated = "K" * 10 + "D" * 10
    mixed = "KD" * 10
    assert kappa(segregated) == pytest.approx(1.0)
    assert 0.0 <= kappa(mixed) < 0.05
    # charge inversion leaves kappa unchanged
    assert kappa(segregated) == pytest.approx(kappa(segregated.replace("K", "J").replace("D", "K").replace("J", "D")))


def test_uncharged_sequence_has_zero_kappa():
    assert kappa("AGS" * 20) == 0.0


def test_rejects_nonstandard_residues():
    from darkmatter.physics import charge as c

    with pytest.raises(ValueError):
        c.charges("PEPTIDX")
    with pytest.raises(ValueError):
        c.charges("")
    with pytest.raises(ValueError):
        c.charges("peptide")  # lowercase -> invalid


def test_fus_statistics_sanity():
    # human FUS P35637: primarily neutral prion-like domain, positive RGG tail
    with open(DATA) as fh:
        fus = json.load(fh)["P35637"]["sequence"].replace("\n", "")
    stats = charge_statistics(fus)
    assert stats["length"] == 526
    assert stats["ncpr"] == pytest.approx(0.0266, abs=5e-4)
    assert stats["fcr"] == pytest.approx(0.1673, abs=5e-4)
    assert stats["kappa"] == pytest.approx(0.1436, abs=5e-4)
    assert stats["scd"] == pytest.approx(0.5388, abs=5e-4)


def test_ncpr_fcr_relations():
    for seq in _sequences()[:30]:
        fp = sum(ch in "KR" for ch in seq) / len(seq)
        fm = sum(ch in "DE" for ch in seq) / len(seq)
        assert fcr(seq) == pytest.approx(fp + fm)
        assert ncpr(seq) == pytest.approx(fp - fm)
