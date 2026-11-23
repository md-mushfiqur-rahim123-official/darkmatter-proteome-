"""Flory-Huggins mean-field phase-coexistence solver (WP-B3).

Free energy of mixing per lattice site for a chain of N residues at
polymer volume fraction phi:

    f(phi) = (phi/N) ln(phi) + (1-phi) ln(1-phi) + chi * phi * (1-phi)

Closed-form results (all verified by tests):
  * Spinodal:          f''(phi) = 0  ->  chi_s(phi) = 0.5*(1/(N phi)+1/(1-phi))
  * Critical point:    phi_c = 1/(1+sqrt(N)),
                       chi_c  = 0.5*(1+1/sqrt(N))**2
                       (the MINIMUM of chi_s over phi -- instability needs
                        chi > chi_s, which first happens at chi_c)
  * Binodal:           common-tangent construction (equality of exchange
                       chemical potential and osmotic pressure between the
                       coexisting phases).

Numerical strategy (validated in tests):
  - The dilute branch of the binodal can lie at astronomically small phi
    (for N=600, chi slightly above chi_c, phi_dilute ~ e**-100 or lower),
    so the dilute root is parametrised in the LOG domain u = ln(phi).
  - For a guessed dilute point phi1, the coexisting dense point phi2 is
    the (unique) root of f'(phi2) = f'(phi1) on (phi_s2, 1); the
    common-tangent residual is chord_slope - tangent_slope.
  - The residual has exactly TWO zeros in the metastable dilute window;
    the LEFTMOST is the true binodal (verified against the exact
    analytical N=1 logit solution), so the scan takes the leftmost
    sign change, hierarchically refined.

Mapping to sequences: WP-B2 calibrates chi = chi(seq) from LLPSDB
phase-diagram data; this module keeps chi abstract so the solver is
independently testable against analytical FH ground truth.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

_EPS = 1e-12


def f_mix(phi: float, chi: float, n: float) -> float:
    """FH free energy per lattice site; phi clipped to (0, 1)."""
    phi = float(np.clip(phi, _EPS, 1 - _EPS))
    return (phi / n) * np.log(phi) + (1 - phi) * np.log(1 - phi) + chi * phi * (1 - phi)


def dfdphi(phi: float, chi: float, n: float) -> float:
    """d f/d phi  (exchange chemical potential / kT up to a constant)."""
    phi = float(np.clip(phi, _EPS, 1 - _EPS))
    return (np.log(phi) + 1.0) / n - np.log(1 - phi) - 1.0 + chi * (1 - 2 * phi)


def chi_spinodal(phi: float, n: float) -> float:
    """chi at the spinodal for a given phi: f''(phi) = 0."""
    phi = float(np.clip(phi, _EPS, 1 - _EPS))
    return 0.5 * (1.0 / (n * phi) + 1.0 / (1.0 - phi))


def critical_point(n: float) -> tuple[float, float]:
    """Analytical FH critical point (phi_c, chi_c)."""
    root_n = np.sqrt(n)
    return 1.0 / (1.0 + root_n), 0.5 * (1.0 + 1.0 / root_n) ** 2


def is_unstable(phi: float, chi: float, n: float) -> bool:
    """Inside the spinodal region (f'' < 0): spontaneous demixing."""
    phi = float(np.clip(phi, _EPS, 1 - _EPS))
    return (1.0 / (n * phi) + 1.0 / (1.0 - phi) - 2.0 * chi) < 0.0


# --- binodal: common tangent ------------------------------------------------
# The dilute binodal branch can sit at phi as small as e^-700, far below the
# clip range of the public f_mix/dfdphi (which exist for robust general use).
# The solver therefore uses f_exact/df_exact: algebraically identical,
# computed in log-domain via log1p so nothing overflows or clips.


def f_exact(phi: float, chi: float, n: float) -> float:
    if phi <= 0 or phi >= 1:
        x = float(np.clip(phi, _EPS, 1 - _EPS))
    else:
        x = phi
    return (x / n) * np.log(x) + (1 - x) * np.log1p(-x) + chi * x * (1 - x)


def df_exact(phi: float, chi: float, n: float) -> float:
    if phi <= 0 or phi >= 1:
        x = float(np.clip(phi, _EPS, 1 - _EPS))
    else:
        x = phi
    return (np.log(x) + 1.0) / n - np.log1p(-x) - 1.0 + chi * (1 - 2 * x)


def _spinodal_roots(chi: float, n: float) -> tuple[float, float]:
    phi_c, _ = critical_point(n)
    ps1 = brentq(lambda p: chi_spinodal(p, n) - chi, _EPS, phi_c,
                 xtol=1e-14, rtol=8.9e-16, maxiter=300)
    ps2 = brentq(lambda p: chi_spinodal(p, n) - chi, phi_c, 1 - _EPS,
                 xtol=1e-14, rtol=8.9e-16, maxiter=300)
    return ps1, ps2


def _coex_pair(chi: float, n: float) -> tuple[float, float] | None:
    """Common-tangent coexistence pair (phi_dilute, phi_dense) for chi > chi_c,
    or None if not found numerically."""
    phi_c, chi_c = critical_point(n)
    if chi <= chi_c:
        return None
    ps1, ps2 = _spinodal_roots(chi, n)
    s_min_right = df_exact(ps2, chi, n)

    def pair_with(phi1: float) -> tuple[float, float]:
        s = df_exact(phi1, chi, n)
        s = max(s, np.nextafter(s_min_right, np.inf))
        phi2 = brentq(lambda p: df_exact(p, chi, n) - s, ps2, 1 - _EPS,
                      xtol=1e-14, rtol=8.9e-16, maxiter=300)
        return phi2, s

    def residual(u: float) -> float:
        phi1 = float(np.exp(u))
        phi2, s = pair_with(phi1)
        chord = (f_exact(phi2, chi, n) - f_exact(phi1, chi, n)) / (phi2 - phi1)
        return chord - s

    # dilute-window bounds in u = ln(phi):
    #   upper bound: the left spinodal root (binodal is left of spinodal)
    #   lower bound: pairing requires f'(phi1) >= f'(phi2-local-min)
    u_hi = np.log(ps1)
    u_lo = -700.0
    if df_exact(_EPS, chi, n) < s_min_right:
        phi_thr = brentq(lambda p: df_exact(p, chi, n) - s_min_right, _EPS, ps1,
                         xtol=1e-14, rtol=8.9e-16, maxiter=300)
        u_lo = np.log(phi_thr) + 1e-12

    # hierarchical scan for the LEFTMOST sign change of the residual:
    # stage 0 over the whole log-window, stage 1 refining the bracket found.
    def scan(us) -> tuple[float, float] | None:
        prev_u, prev_y = None, None
        for u in us:
            try:
                y = residual(float(u))
            except ValueError:
                prev_u, prev_y = float(u), None
                continue
            if y != y:  # NaN
                prev_u, prev_y = float(u), None
                continue
            if prev_y is not None:
                if (prev_y >= 0.0 and y < 0.0) or (prev_y < 0.0 and y > 0.0):
                    return (prev_u, float(u))
                if prev_y == 0.0:
                    return (prev_u, prev_u)
            if y == 0.0:
                return (float(u), float(u))
            prev_u, prev_y = float(u), y
        return None

    bracket = scan(np.linspace(u_lo, u_hi, 240))
    if bracket is None:
        return None
    if bracket[0] != bracket[1]:
        bracket = scan(np.linspace(bracket[0], bracket[1], 4000))
        if bracket is None:
            return None

    if bracket is None:
        return None
    if bracket[0] == bracket[1]:
        # exact hit on a grid node
        u_star = bracket[0]
        phi1 = float(np.exp(u_star))
    else:
        u_star = brentq(residual, bracket[0], bracket[1], xtol=1e-13,
                        rtol=8.9e-16, maxiter=300)
        phi1 = float(np.exp(u_star))
    phi2, _ = pair_with(phi1)
    return phi1, phi2


def binodal(chi_grid: np.ndarray | list[float], n: float) -> list[dict]:
    """Common-tangent binodal for each chi. Returns a list of
    {"chi", "phi_dilute", "phi_dense", "coexistence"} entries, in the
    SAME ORDER as the input grid. No coexistence where chi <= chi_c."""
    out = []
    for chi in np.atleast_1d(np.asarray(chi_grid, dtype=float)):
        got = _coex_pair(float(chi), n)
        if got is None:
            out.append({"chi": float(chi), "phi_dilute": None,
                        "phi_dense": None, "coexistence": False})
        else:
            out.append({"chi": float(chi), "phi_dilute": got[0],
                        "phi_dense": got[1], "coexistence": True})
    return out


def csat(chi: float, n: float) -> float | None:
    """Saturation (dilute-branch binodal) concentration for a given chi.
    Monotonically decreasing in chi: higher chi -> lower csat."""
    r = binodal([chi], n)[0]
    return r["phi_dilute"] if r["coexistence"] else None


def solve_binodal_for_phi(phi_target: float, n: float, chi_lo: float = 0.0,
                          chi_hi: float = 8.0) -> float | None:
    """Inverse problem: find chi such that the dilute binodal branch equals
    phi_target (used by WP-B2 to calibrate chi from a measured csat).

    The residual phi_target - csat(chi) is non-monotone (it is positive in
    the mixed regime below chi_c AND at very large chi where csat dives
    below float range), so we scan a log-spaced chi grid for the bracket
    and refine the LOWEST-chi root with brentq.
    """
    import numpy as _np
    _, chi_c0 = critical_point(n)
    grid = _np.geomspace(max(chi_lo, chi_c0 * 1.0000001), chi_hi, 48)
    prev_chi, prev_r = None, None
    for chi in grid:
        val = _resid_one(float(chi), phi_target, n)
        if prev_r is not None and prev_r * val <= 0.0 and val != prev_r:
            return brentq(lambda c: _resid_one(c, phi_target, n),
                          float(prev_chi), float(chi), xtol=1e-11, maxiter=80)
        prev_chi, prev_r = float(chi), val
    return None


def _resid_one(chi: float, phi_target: float, n: float) -> float:
    if chi <= critical_point(n)[1]:
        return phi_target - 1e-12  # mixed phase: dilute branch ~ 0
    r = binodal([chi], n)[0]
    if not r["coexistence"]:
        return phi_target - 1e-12
    return phi_target - r["phi_dilute"]
