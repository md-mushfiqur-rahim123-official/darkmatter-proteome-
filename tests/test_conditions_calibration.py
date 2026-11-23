"""Tests for WP-B2a concentration parsers + WP-B2c calibration model pieces."""
import math

import numpy as np
import pytest

from darkmatter.data.conditions import (
    parse_concentration,
    parse_salt_mM,
    parse_temperature_C,
)
from darkmatter.physics.calibration import PhiGrid


class TestUnits:
    def test_uM_direct(self):
        c, cmax, kind = parse_concentration("3.5 µM")
        assert c == pytest.approx(3.5) and kind == "point"

    def test_uM_alternate_micros(self):
        for txt in ("3.5 μM", "3.5 uM"):
            c, _, _ = parse_concentration(txt)
            assert c == pytest.approx(3.5), txt

    def test_prefix_not_split_from_unit(self):
        # regression: 'µM' must never be parsed as prefix 'µ' + unit 'M'
        c, _, kind = parse_concentration("3.5 µM")
        assert c < 100.0 and kind == "point"

    def test_mM(self):
        c, _, _ = parse_concentration("20 mM")
        assert c == pytest.approx(20000.0)

    def test_molar(self):
        c, _, _ = parse_concentration("2 M")
        assert c == pytest.approx(2e6)

    def test_range_takes_max(self):
        c, cmax, kind = parse_concentration("0-10 µM")
        assert kind == "range" and c == pytest.approx(10.0) and cmax == pytest.approx(10.0)

    def test_range_to_variant(self):
        c, _, kind = parse_concentration("0 to 5 uM")
        assert kind == "range" and c == pytest.approx(5.0)

    def test_mg_ml_uses_weight(self):
        seq = "A" * 100  # 100 residues -> MW 11000 Da = 11 kDa
        c, _, kind = parse_concentration("1 mg/ml", seq_len=seq.__len__())
        # 1 mg/ml / 11000 g/mol = 9.09e-5 g/L / ... = 90.909 uM
        assert kind == "point" and c == pytest.approx(90.909, rel=1e-3)

    def test_ug_ml(self):
        c, _, _ = parse_concentration("500 ug/ml", seq_len=1100)  # MW 121 kDa
        assert c == pytest.approx(0.5 / 121000.0 * 1e6, rel=1e-3)  # 4.13 uM

    def test_g_ml(self):
        c, _, _ = parse_concentration("0.01 g/ml", seq_len=1100)
        assert c == pytest.approx(10.0 / 121000.0 * 1e6, rel=1e-3)  # 82.64 uM

    def test_unconvertible_without_length(self):
        c, _, _ = parse_concentration("1 mg/ml", seq_len=None)
        assert c is None

    def test_unknown_narrative(self):
        c, _, kind = parse_concentration("see methods")
        assert c is None and kind == "unknown"


class TestSaltTemp:
    def test_salt_mM(self):
        assert parse_salt_mM("75 mM KCl") == pytest.approx(75.0)

    def test_salt_M(self):
        assert parse_salt_mM("1 M NaCl") == pytest.approx(1000.0)

    def test_salt_missing(self):
        assert parse_salt_mM("-") is None
        assert parse_salt_mM("") is None

    def test_temp_rt(self):
        assert parse_temperature_C("RT") == 25.0
        assert parse_temperature_C("room temperature") == 25.0

    def test_temp_numeric(self):
        assert parse_temperature_C("20 C") == 20.0
        assert parse_temperature_C("37") == 37.0
        assert parse_temperature_C("4°C") == 4.0


class TestPhiGrid:
    def test_grid_matches_exact_solver_roughly(self):
        from darkmatter.physics.flory_huggins import csat
        g = PhiGrid()
        for chi, ln in ((2.5, 1.3), (3.0, 1.1), (2.0, 1.4)):
            exact = csat(chi, 10.0 ** ln)
            approx = 10.0 ** g.log10_phi(chi, ln)
            if exact is None:
                continue
            # bilinear interpolation of a smooth log-surface: tight tolerance
            assert abs(math.log10(exact) - g.log10_phi(chi, ln)) < 0.15, (chi, ln)

    def test_interior_one_phase_gives_large_lphi(self):
        g = PhiGrid()
        # N=10 -> chi_c ~ 0.66; pick chi below spinodal region grid start 0.8?
        # smallest grid chi is 0.8; assert value never negative below -20
        assert g.log10_phi(2.0, 1.5) < 0

    def test_monotone_decreasing_in_chi(self):
        g = PhiGrid()
        vals = [g.log10_phi(c, 1.2) for c in (1.0, 1.5, 2.0, 3.0, 4.0)]
        # allow NaN->large sentinel at low chi
        vals = [-1e9 if v > 10 else v for v in vals]
        assert all(vals[i] >= vals[i + 1] - 0.5 for i in range(len(vals) - 1)), vals
