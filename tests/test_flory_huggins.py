"""Ground-truth tests for the Flory-Huggins solver (analytical results only)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from darkmatter.physics.flory_huggins import (
    binodal, chi_spinodal, critical_point, csat, df_exact, dfdphi, f_mix,
    f_exact, is_unstable, solve_binodal_for_phi,
)


def test_critical_point_analytical():
    # chi_c is the MINIMUM of the spinodal curve chi_s(phi) over phi
    # (instability needs chi > chi_s, and the mixed state first becomes
    # unstable at the lowest point of the spinodal), attained at phi_c.
    for n in (1.0, 4.0, 50.0, 600.0):
        phi_c, chi_c = critical_point(n)
        grid = np.linspace(0.002, 0.998, 300001)
        vals = 0.5 * (1.0 / (n * grid) + 1.0 / (1.0 - grid))
        imin = int(np.argmin(vals))
        assert abs(grid[imin] - phi_c) < 1e-4, (n, grid[imin], phi_c)
        assert abs(vals[imin] - chi_c) < 1e-4, (n, vals[imin], chi_c)


def test_spinodal_instability_sign():
    n = 100.0
    _, chi_c = critical_point(n)
    assert is_unstable(0.5, chi_c * 2, n)
    assert not is_unstable(0.5, chi_c * 0.5, n)


def test_fmix_derivatives_consistency():
    n, chi = 40.0, 0.8
    for phi in (0.05, 0.2, 0.5, 0.8, 0.95):
        num = (f_mix(phi + 1e-7, chi, n) - f_mix(phi - 1e-7, chi, n)) / 2e-7
        assert abs(num - dfdphi(phi, chi, n)) < 1e-4


def test_binodal_reduces_to_symmetry_n1():
    # For N=1, chi_c = 2 and FH is symmetric: coexisting phases must be
    # mirror images; also no coexistence below chi_c (sanity-checked here).
    assert not any(r["coexistence"] for r in binodal([0.8, 1.5], n=1.0))
    grid = [2.5, 3.0, 4.0]
    res = binodal(grid, n=1.0)
    for r in res:
        assert r["coexistence"]
        assert abs((r["phi_dilute"] + r["phi_dense"]) - 1.0) < 1e-6


def test_binodal_common_tangent_stationary():
    # Verify the actual common-tangent conditions at returned points.
    # Use the log-domain exact forms (mathematically identical; the clipped
    # f_mix/dfdphi would clamp phi down to 1e-12 and corrupt the dilute branch).
    n = 600.0
    res = binodal([1.2, 2.0, 3.0], n=n)
    for r in res:
        if not r["coexistence"]:
            continue
        p1, p2 = r["phi_dilute"], r["phi_dense"]
        s1 = df_exact(p1, r["chi"], n)
        s2 = df_exact(p2, r["chi"], n)
        chord = (f_exact(p2, r["chi"], n) - f_exact(p1, r["chi"], n)) / (p2 - p1)
        assert abs(s1 - s2) < 1e-6
        assert abs(chord - s1) < 1e-6
        assert p1 < critical_point(n)[0] < p2


def test_binodal_reaches_critical_point():
    n = 600.0
    phi_c, chi_c = critical_point(n)
    for delta in (1e-6, 1e-4):
        res = binodal([chi_c + delta], n=n)
        assert res[0]["coexistence"], (delta, res)
        # near-critical coexistence points hug phi_c
        assert abs(res[0]["phi_dilute"] - phi_c) < 0.05 or \
            abs(res[0]["phi_dense"] - phi_c) < 0.09


def test_csat_monotone_decreasing_in_chi():
    # NOTE on N: with the textbook entropy term phi/N ln(phi), large N
    # (full chain length) pushes the dilute binodal to absurdly small phi
    # (csat ~ e^{-N const}), beyond float64 for chi >> chi_c. Real IDP
    # modelling therefore uses an EFFECTIVE blob/sticker length N_eff <<
    # N, calibrated per sequence (WP-B2). Tests here use N_eff in the
    # numerically meaningful range.
    n = 25.0
    _, chi_c = critical_point(n)
    chis = [chi_c * 1.05, chi_c * 1.2, chi_c * 1.5, chi_c * 2.0]
    vals = [csat(c, n) for c in chis]
    assert all(v is not None for v in vals)
    assert all(a > b for a, b in zip(vals, vals[1:]))


@pytest.mark.parametrize("n", [10.0, 25.0, 300.0])
def test_inverse_calibration_roundtrip(n):
    _, chi_c = critical_point(n)
    mult = 1.02 if n >= 300 else 1.5  # keep csat within float precision
    chi = chi_c * mult
    phi1 = csat(chi, n)
    assert phi1 is not None
    chi_back = solve_binodal_for_phi(phi1, n)
    assert chi_back is not None
    assert abs(chi_back - chi) < 1e-4
