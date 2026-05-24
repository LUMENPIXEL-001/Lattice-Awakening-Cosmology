# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LUMENPIXEL
#
# Released under the MIT License.

"""
LVC final-v1.5 -- Single-file reproduction script
==================================================

Reproduces the numerical results in the final-v1.5 working paper:

    "Lagrangian Variable Cosmology final-v1.5:
     Implementation-Consistent Reproduction of the v1.1 Modulation
     and a Three-Rung Ladder Extension within the v1.1 Lagrangian"
                                LUMENPIXEL, May 2026

Single self-contained Python file. Required external dependencies:
numpy, pandas, scipy.

BUG-CORRECTION NOTE
-------------------
The reproduction scripts of v0.1 through v1.4 carried an inconsistency
between the BAO/SN likelihood blocks (paper-intent: H = H_LCDM * T) and
the Planck distance-prior block (T applied to dark-energy term only).

  buggy   (v0.1-v1.4 _R_lA_raw):
      H_E^2 = H0^2 * [ Om*(1+z)^3 + Or*(1+z)^4
                       + (1 - Om - Or) * T(z)^2 ]   <-- T on DE only

  paper-intent (this v1.5 script):
      H_E^2 = H0^2 * [ Om*(1+z)^3 + Or*(1+z)^4 + (1-Om-Or) ] * T(z)^2

The two coincide for z << z* but diverge in the z*-integration that
controls D_M(z*) in the distance-prior likelihood. The v0.1-v1.4
chi^2/dBIC numbers carry approximately 18.5 units of inflation
(PP-included) from this inconsistency. The present script uses
paper-intent everywhere; v1.4 numbers are reproduced in their
ORIGINAL (buggy) form only in Section 8 for documentation, with both
buggy and corrected values printed side-by-side.

The v1.5 three-rung ladder construction is implemented in Section 9
and is the principal new content of this script.

Reproduces (key paper numbers, PP-included unless stated):
----------------------------------------------------------
Section 8:  Bug-effect tables (Tables 1 and 2 of the paper)
    LCDM PP-incl       (k=3)   chi^2 = 1624.85
    v1.1 buggy         (k=2)   chi^2 = 1601.999   dBIC = -30.31
    v1.1 paper-intent  (k=2)   chi^2 = 1620.49    dBIC = -11.82
    v1.4 buggy         (k=2)   chi^2 = 1601.96    dBIC = -30.34
    v1.4 paper-intent  (k=2)   chi^2 = 1620.62    dBIC = -11.69

Section 9:  Three-rung ladder fits (Tables 3-7 of the paper)
    Free A_2 fit (k=3)         A_2 best = -0.0531 (3.05 sigma from 0)
    n_max=2 r=phi locked (k=2) chi^2 = 1615.37    dBIC = -16.94
    n_max=4 r=phi locked (k=2) chi^2 = 1595.26    dBIC = -37.04
    3-rung paper construction  chi^2 = 1590.69    dBIC = -41.83  (k=2)
        A_2/A_1 = -1, A_3/A_1 = -phi
        H0 = 69.91, Omega_b h^2 = 0.02249, r_d = 144.52

Section 10: f sigma_8 fit (Tables 8-9 of the paper)
    LCDM at Om-lock            chi^2 = 12.18, sigma_8 = 0.811
    3-rung, alpha_g = 0        chi^2 = 8.00,  sigma_8 = 0.834
    3-rung, alpha_g = -5 A_1   chi^2 = 13.53, sigma_8 = 0.813

Section 11: Joint background + fsigma8 (Table 10 of the paper)
    PP-included (N=1754)       chi^2 = 1604.22, dBIC = -40.27
    PP-excluded (N=53)         chi^2 =   65.85, dBIC = -29.27

External data
-------------
Pantheon+ data files needed for PP-included sections:
    Pantheon+SH0ES.dat
    Pantheon+SH0ES_STAT+SYS.cov

If absent, the script offers automatic git-based download. PP-excluded
sections (N=37) do not require these files.

Usage
-----
    python lvc_v15_reproduce.py                       # default (full run)
    python lvc_v15_reproduce.py --skip-pp-included    # PP-excluded only
    python lvc_v15_reproduce.py --skip-fsigma8        # skip Section 10
    python lvc_v15_reproduce.py --help                # full options

License
-------
MIT License. Copyright (c) 2026 LUMENPIXEL.
"""

import os
import sys
import time
import json
import argparse
import subprocess
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import quad, solve_ivp
from scipy.optimize import differential_evolution, minimize, brentq


# ============================================================================
# SECTION 1.  CONSTANTS
# ============================================================================

C_KMS         = 299_792.458
THETA_STAR    = 0.010409
SIG_THETA_STAR = 3.1e-5
Z_STAR        = 1090.0
H0_SHOES      = 73.04
SIG_H0_SHOES  = 1.04

OMEGA_GAMMA_H2 = 2.473e-5
NEFF_STD       = 3.046
OMEGA_R_H2     = OMEGA_GAMMA_H2 * (1 + 7/8 * (4/11)**(4/3) * NEFF_STD)

PI  = np.pi
E   = np.e
PHI = (1 + np.sqrt(5)) / 2.0
SQRT5 = np.sqrt(5)
INV_PHI    = 1.0 / PHI
THETA_C_PHI = np.log(PHI)
LNPHI_3 = THETA_C_PHI ** 3
LNPHI_4 = THETA_C_PHI ** 4

# v1.1 recursive Omega_m lock: Om = 1/(1 + e^{3 ln phi / phi}) = 0.290653
OM_RECURSIVE = 1.0 / (1.0 + np.exp(3.0 * THETA_C_PHI / PHI))

# Planck 2018 distance prior central values (Chen, Huang, Wang 2019)
R_PLANCK   = 1.7493
LA_PLANCK  = 301.462
WB_PLANCK  = 0.02236
SIG_R, SIG_LA, SIG_WB = 0.0049, 0.090, 0.00015
CORR_DP = np.array([
    [ 1.0000,  0.4720, -0.6536],
    [ 0.4720,  1.0000, -0.3392],
    [-0.6536, -0.3392,  1.0000]])
SIG_DP = np.array([SIG_R, SIG_LA, SIG_WB])
COV_DP = CORR_DP * SIG_DP[:, None] * SIG_DP[None, :]
COVINV_DP = np.linalg.inv(COV_DP)

# Data set sizes
N_PP_EXCLUDED = 37
N_PP_INCLUDED = 1701 + 13 + 11 + 2 + 7 + 3 + 1   # = 1738

# Reference Sigma_8 (Planck 2018 primary)
SIGMA8_PLANCK = 0.811


# ============================================================================
# SECTION 2.  PANTHEON+ DATA LOADER (auto-download)
# ============================================================================

PP_DIR = os.environ.get(
    "PANTHEONPLUS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "PP_data"))
PP_DAT = os.path.join(PP_DIR, "Pantheon+SH0ES.dat")
PP_COV = os.path.join(PP_DIR, "Pantheon+SH0ES_STAT+SYS.cov")


def auto_download_pp():
    print(f"\nPantheon+ data not found at {PP_DIR}.")
    print("Attempting automatic download from GitHub...")
    response = input("\nDownload now? [Y/n] ").strip().lower()
    if response and response != 'y':
        print("Aborted. Set PANTHEONPLUS_DIR or sparse-clone manually:")
        print("  git clone --depth=1 --filter=blob:none --no-checkout \\")
        print("      https://github.com/PantheonPlusSH0ES/DataRelease.git")
        print("  cd DataRelease")
        print("  git sparse-checkout init --cone")
        print("  git sparse-checkout set 'Pantheon+_Data/4_DISTANCES_AND_COVAR'")
        print("  git checkout main")
        sys.exit(1)
    os.makedirs(PP_DIR, exist_ok=True)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cmds = [
            ['git', 'clone', '--depth=1', '--filter=blob:none', '--no-checkout',
             'https://github.com/PantheonPlusSH0ES/DataRelease.git', 'DataRelease'],
            ['git', '-C', 'DataRelease', 'sparse-checkout', 'init', '--cone'],
            ['git', '-C', 'DataRelease', 'sparse-checkout', 'set',
             'Pantheon+_Data/4_DISTANCES_AND_COVAR'],
            ['git', '-C', 'DataRelease', 'checkout', 'main'],
        ]
        for cmd in cmds:
            print("  Running:", " ".join(cmd))
            r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True,
                               timeout=180)
            if r.returncode != 0:
                print("ERROR:", r.stderr); sys.exit(1)
        src = os.path.join(tmp, 'DataRelease', 'Pantheon+_Data',
                           '4_DISTANCES_AND_COVAR')
        for fname in ['Pantheon+SH0ES.dat', 'Pantheon+SH0ES_STAT+SYS.cov']:
            with open(os.path.join(src, fname), 'rb') as fi, \
                 open(os.path.join(PP_DIR, fname), 'wb') as fo:
                fo.write(fi.read())
            print(f"  Copied {fname}")
    print("Download complete.\n")


def load_pantheonplus():
    if not os.path.exists(PP_DAT) or not os.path.exists(PP_COV):
        auto_download_pp()
    df = pd.read_csv(PP_DAT, sep=r'\s+')
    is_calib = (df['IS_CALIBRATOR'].values == 1)
    out = dict(
        zHD     = df['zHD'].values,
        zHEL    = df['zHEL'].values,
        mB      = df['m_b_corr'].values,
        ceph    = df['CEPH_DIST'].values,
        is_calib= is_calib,
        N       = len(df),
    )
    with open(PP_COV) as f:
        first = f.readline().strip()
    Ncov = int(first)
    cov = np.loadtxt(PP_COV, skiprows=1).reshape((Ncov, Ncov))
    out['Cinv'] = np.linalg.inv(cov)
    return out


# ============================================================================
# SECTION 3.  BACKGROUND COSMOLOGY AND MODULATION T(z)
# ============================================================================

_Z_GRID = np.concatenate([[0.0], np.geomspace(1e-3, 5.0, 200)])


def E_lcdm(z, Om):
    """Standard flat LCDM dimensionless Hubble rate, ignoring radiation
    (used in BAO/SN distance-grid; radiation only matters for z >> 1)."""
    return np.sqrt(Om*(1+z)**3 + (1-Om))


# ---------------------------------------------------------------------------
# Modulation functions
# ---------------------------------------------------------------------------

def T_v11(z, A=LNPHI_4):
    """v1.1 single rung:  T(z) = 1 + A sech^2((theta - theta_c)/w),
    theta = ln(1+z),  theta_c = ln phi,  w = (ln phi)^3.
    """
    z = np.asarray(z, dtype=float)
    theta = np.log(1.0 + z)
    arg = np.clip((theta - THETA_C_PHI) / LNPHI_3, -50.0, 50.0)
    return 1.0 + A / np.cosh(arg) ** 2


def T_v14(z, alpha_b=LNPHI_4):
    """v1.4 exponential lift:
       T(z) = exp[ alpha_b sech^2((theta - theta_c)/w) ].
    Default alpha_b = A = (ln phi)^4 (v1.1 small-amplitude match).
    """
    z = np.asarray(z, dtype=float)
    theta = np.log(1.0 + z)
    arg = np.clip((theta - THETA_C_PHI) / LNPHI_3, -50.0, 50.0)
    return np.exp(alpha_b / np.cosh(arg) ** 2)


def T_ladder(z, amps, positions=None, w=LNPHI_3):
    """v1.3/v1.5 multi-rung sech^2 ladder:
       T(z) - 1 = sum_n A_n sech^2((theta - theta_c,n)/w)
    positions default to {n ln phi}_{n=1..len(amps)}.
    """
    amps = np.asarray(amps, dtype=float)
    if positions is None:
        positions = np.array([(n+1) * THETA_C_PHI for n in range(len(amps))])
    else:
        positions = np.asarray(positions, dtype=float)
    z = np.asarray(z, dtype=float)
    theta = np.log(1.0 + z)
    t = np.ones_like(theta)
    for i in range(len(amps)):
        arg = np.clip((theta - positions[i]) / w, -50.0, 50.0)
        t = t + amps[i] / np.cosh(arg)**2
    return t


# v1.5 paper construction: three-rung lock pattern
LADDER_V15_AMPS = np.array([+LNPHI_4, -LNPHI_4, -PHI * LNPHI_4])
LADDER_V15_POS  = np.array([THETA_C_PHI, 2*THETA_C_PHI, 3*THETA_C_PHI])


def T_v15(z):
    """v1.5 three-rung lock pattern:
       A_1 = (ln phi)^4
       A_2 = -A_1
       A_3 = -phi * A_1
       theta_c,n = n ln phi,  w = (ln phi)^3
    """
    return T_ladder(z, LADDER_V15_AMPS, LADDER_V15_POS, LNPHI_3)


# ---------------------------------------------------------------------------
# Comoving distance grid -- BAO/SN form (paper-intent: H = H_LCDM * T)
# ---------------------------------------------------------------------------

def comoving_dist_grid(H0, Om, T_func=None):
    """D_C(z) on _Z_GRID. T_func=None for Lambda-CDM, else callable."""
    if T_func is None:
        H = H0 * E_lcdm(_Z_GRID, Om)
    else:
        Tv = T_func(_Z_GRID)
        if np.min(Tv) < 0.4 or np.max(Tv) > 2.0:
            return None    # reject pathological values
        H = H0 * E_lcdm(_Z_GRID, Om) * Tv
    # Cumulative trapezoid integration of c/H from 0 to z
    dz = np.diff(_Z_GRID)
    inv_H = C_KMS / H
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (inv_H[:-1] + inv_H[1:]) * dz)])
    return cum


# ============================================================================
# SECTION 4.  PANTHEON+ LIKELIHOOD
# ============================================================================

def chi2_panplus(H0, Om, T_func, pp):
    """Pantheon+ + SH0ES Cepheid host distance ladder, with analytic
    marginalisation over the absolute magnitude M_hat."""
    DC = comoving_dist_grid(H0, Om, T_func)
    if DC is None:
        return 1e10, 0.0
    DM_at_zHD = np.interp(pp['zHD'], _Z_GRID, DC)
    mu_pred = np.where(
        pp['is_calib'],
        pp['ceph'],
        5*np.log10((1+pp['zHEL']) * DM_at_zHD) + 25)
    d = pp['mB'] - mu_pred
    ones = np.ones(pp['N'])
    Cinv = pp['Cinv']
    Cinv_d = Cinv @ d
    Cinv_1 = Cinv @ ones
    s11 = ones @ Cinv_1
    s1d = ones @ Cinv_d
    M_hat = s1d / s11
    r = d - M_hat
    return float(r @ Cinv @ r), float(M_hat)


# ============================================================================
# SECTION 5.  BAO DATA
# ============================================================================

DESI_DR2_DV = [(0.295, 7.942, 0.075)]
DESI_DR2_PAIR = [
    ('LRG1', 0.510, 13.587, 0.169, 21.863, 0.427, -0.475),
    ('LRG2', 0.706, 17.347, 0.180, 19.458, 0.332, -0.423),
    ('LRG3', 0.934, 21.574, 0.153, 17.641, 0.193, -0.425),
    ('QSO',  1.321, 27.605, 0.320, 14.178, 0.217, -0.437),
    ('ELG2', 1.484, 30.519, 0.758, 12.816, 0.513, -0.489),
    ('Lya',  2.330, 38.988, 0.531,  8.632, 0.101, -0.431),
]
DESI_DR1_DV = [(0.295, 7.93, 0.150)]
DESI_DR1_PAIR = [
    ('LRG1', 0.510, 13.62, 0.25, 20.98, 0.61, -0.445),
    ('LRG2', 0.706, 16.85, 0.32, 20.08, 0.60, -0.420),
    ('LRG3', 0.930, 21.71, 0.28, 17.88, 0.35, -0.389),
    ('ELG2', 1.317, 27.79, 0.69, 13.82, 0.42, -0.444),
    ('Lya',  2.330, 39.71, 0.94,  8.52, 0.17, -0.477),
]
BOSS_DM = [(0.38, 10.27, 0.15)]
BOSS_DH = [(0.38, 24.89, 0.58, -0.42)]
EBOSS_DM_PAIRS = [
    ('eBOSS_LRG', 0.698, 17.86, 0.33, 19.33, 0.53, -0.40),
    ('eBOSS_QSO', 1.480, 30.21, 0.79, 13.23, 0.47, -0.40),
    ('eBOSS_Lya', 2.334, 37.60, 1.90,  8.93, 0.28, -0.45),
]
EBOSS_DV = [('eBOSS_ELG', 0.845, 18.33, 0.57)]


# ============================================================================
# SECTION 6.  DISTANCE PRIOR LIKELIHOOD  (PAPER-INTENT)
# ============================================================================
#
# CRITICAL BUG FIX -- v0.1 through v1.4 reproduction scripts used:
#
#     H_E^2 = H0^2 * [ Om*(1+z)^3 + Or*(1+z)^4
#                      + (1 - Om - Or) * T(z)^2 ]      <-- INCORRECT
#                                                          (T on DE only)
#
# The paper-stated form is H = H_LCDM * T everywhere, equivalently:
#
#     H_E^2 = H0^2 * [ Om*(1+z)^3 + Or*(1+z)^4
#                      + (1 - Om - Or) ] * T(z)^2     <-- PAPER-INTENT
#
# The two coincide for z << z* (where the DE term is non-negligible) but
# diverge for the radiation+matter-dominated integration up to z* ~ 1090.
# The buggy form effectively decouples the distance prior from T(z) at
# high z, freeing the joint optimiser to choose (H0, Om) that suit BAO/SN
# without paying a DP cost.
#
# This script uses ONLY the paper-intent form. The buggy form is
# implemented separately in Section 8 (function _R_lA_raw_BUGGY) for
# reproduction of the v1.1/v1.4 paper numbers.
# ============================================================================

def _z_star_HS(Ob_h2, Om_h2):
    """Hu-Sugiyama 1996 fitting formula for the photon-baryon decoupling
    redshift z_*. Standard input to compressed Planck distance priors."""
    g1 = 0.0783 * Ob_h2**(-0.238) / (1 + 39.5 * Ob_h2**0.763)
    g2 = 0.560 / (1 + 21.1 * Ob_h2**1.81)
    return 1048 * (1 + 0.00124 * Ob_h2**(-0.738)) * (1 + g1 * Om_h2**g2)


def _R_lA_raw(H0, Om, Ob_h2, T_func=None):
    """Raw R, lA, D_M(z*), r_s(z*) without calibration.  PAPER-INTENT
    (this is the FIXED version; see _R_lA_raw_BUGGY in Section 8 for the
    v0.1-v1.4 inconsistent form used to reproduce historical numbers).

    Paper-intent:  H_E^2 = H_LCDM^2(z) * T(z)^2   (uniform T everywhere)
    """
    h = H0 / 100.0
    Om_h2 = Om * h * h
    z_star = _z_star_HS(Ob_h2, Om_h2)
    Or_frac = OMEGA_R_H2 / h**2
    if T_func is None:
        def H_E(z):
            return H0 * np.sqrt(Om*(1+z)**3 + Or_frac*(1+z)**4
                                + (1 - Om - Or_frac))
    else:
        def H_E(z):
            T_val = float(T_func(np.array([z]))[0])
            H_lcdm_sq = (Om*(1+z)**3 + Or_frac*(1+z)**4
                         + (1 - Om - Or_frac))
            return H0 * np.sqrt(H_lcdm_sq) * T_val   # <-- paper-intent
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        DM_star, _ = quad(lambda z: C_KMS/H_E(z), 0, z_star, limit=400)
        rs, _ = quad(
            lambda z: 1.0/np.sqrt(3*(1 + 3*Ob_h2/(4*OMEGA_GAMMA_H2)/(1+z)))
                      * C_KMS/H_E(z),
            z_star, 1e6, limit=400)
    R_raw = np.sqrt(Om_h2) * DM_star * 100.0 / C_KMS
    lA_raw = PI * DM_star / rs
    return R_raw, lA_raw, DM_star, rs, z_star


# Calibration: at Planck fiducial flat LCDM, multiplicative factor.
# This is identical under buggy and fixed conventions (T = None branch).
_R_AT_PLANCK, _LA_AT_PLANCK, _, _, _ = _R_lA_raw(67.36, 0.3153, 0.02237, None)
CAL_R  = R_PLANCK  / _R_AT_PLANCK
CAL_LA = LA_PLANCK / _LA_AT_PLANCK


def chi2_DP(H0, Om, Ob_h2, T_func=None):
    """Compressed Planck 2018 distance prior chi^2 (paper-intent)."""
    R_raw, lA_raw, _, _, _ = _R_lA_raw(H0, Om, Ob_h2, T_func)
    R_cal = R_raw * CAL_R
    lA_cal = lA_raw * CAL_LA
    x = np.array([R_cal, lA_cal, Ob_h2])
    xref = np.array([R_PLANCK, LA_PLANCK, WB_PLANCK])
    d = x - xref
    return float(d @ COVINV_DP @ d), float(R_cal), float(lA_cal)


def DM_at_zstar(H0, Om, Ob_h2, T_func=None):
    """D_M(z*) used to derive r_d geometrically via theta_* = r_d / D_M(z*)."""
    _, _, DM_star, _, _ = _R_lA_raw(H0, Om, Ob_h2, T_func)
    return DM_star


# ============================================================================
# SECTION 7.  COMBINED LIKELIHOODS
# ============================================================================

def chi2_bao_combined(H0, Om, T_func, rd):
    """Combined DR1 + DR2 + BOSS + eBOSS BAO chi^2.
    BOSS DR12 DM and DH are correlated (corr = -0.42) and enter via a 2x2 cov."""
    DC = comoving_dist_grid(H0, Om, T_func)
    if DC is None:
        return 1e10
    if T_func is None:
        H = H0 * E_lcdm(_Z_GRID, Om)
    else:
        H = H0 * E_lcdm(_Z_GRID, Om) * T_func(_Z_GRID)
    chi2 = 0.0
    for z, DVo, sig in DESI_DR2_DV:
        DM = np.interp(z, _Z_GRID, DC); Hv = np.interp(z, _Z_GRID, H)
        DV = (z * DM**2 * C_KMS / Hv) ** (1/3)
        chi2 += ((DVo - DV / rd) / sig) ** 2
    for label, z, DMo, sDM, DHo, sDH, corr in DESI_DR2_PAIR:
        DMp = np.interp(z, _Z_GRID, DC) / rd
        DHp = C_KMS / np.interp(z, _Z_GRID, H) / rd
        cov = np.array([[sDM**2, corr * sDM * sDH],
                        [corr * sDM * sDH, sDH**2]])
        d = np.array([DMo - DMp, DHo - DHp])
        chi2 += d @ np.linalg.inv(cov) @ d
    for z, DVo, sig in DESI_DR1_DV:
        DM = np.interp(z, _Z_GRID, DC); Hv = np.interp(z, _Z_GRID, H)
        DV = (z * DM**2 * C_KMS / Hv) ** (1/3)
        chi2 += ((DVo - DV / rd) / sig) ** 2
    for label, z, DMo, sDM, DHo, sDH, corr in DESI_DR1_PAIR:
        DMp = np.interp(z, _Z_GRID, DC) / rd
        DHp = C_KMS / np.interp(z, _Z_GRID, H) / rd
        cov = np.array([[sDM**2, corr * sDM * sDH],
                        [corr * sDM * sDH, sDH**2]])
        d = np.array([DMo - DMp, DHo - DHp])
        chi2 += d @ np.linalg.inv(cov) @ d
    # BOSS DR12: DM at z=0.38 with sDM=0.15, DH at z=0.38 with sDH=0.58,
    # correlation -0.42  -- enter via a 2x2 cov
    for (z, DMo, sDM), (z2, DHo, sDH, corr) in zip(BOSS_DM, BOSS_DH):
        DMp = np.interp(z, _Z_GRID, DC) / rd
        DHp = C_KMS / np.interp(z, _Z_GRID, H) / rd
        cov = np.array([[sDM**2, corr * sDM * sDH],
                        [corr * sDM * sDH, sDH**2]])
        d = np.array([DMo - DMp, DHo - DHp])
        chi2 += d @ np.linalg.inv(cov) @ d
    for label, z, DMo, sDM, DHo, sDH, corr in EBOSS_DM_PAIRS:
        DMp = np.interp(z, _Z_GRID, DC) / rd
        DHp = C_KMS / np.interp(z, _Z_GRID, H) / rd
        cov = np.array([[sDM**2, corr * sDM * sDH],
                        [corr * sDM * sDH, sDH**2]])
        d = np.array([DMo - DMp, DHo - DHp])
        chi2 += d @ np.linalg.inv(cov) @ d
    for label, z, DVo, sig in EBOSS_DV:
        DM = np.interp(z, _Z_GRID, DC); Hv = np.interp(z, _Z_GRID, H)
        DV = (z * DM**2 * C_KMS / Hv) ** (1/3)
        chi2 += ((DVo - DV / rd) / sig) ** 2
    return float(chi2)


def chi2_full(H0, Om, Ob_h2, T_func, pp):
    """PP + BAO + SH0ES + DP."""
    DM_star = DM_at_zstar(H0, Om, Ob_h2, T_func)
    rd = THETA_STAR * DM_star
    chi_pp, _ = chi2_panplus(H0, Om, T_func, pp)
    chi_bao = chi2_bao_combined(H0, Om, T_func, rd)
    chi_h = ((H0_SHOES - H0) / SIG_H0_SHOES) ** 2
    chi_dp, _, _ = chi2_DP(H0, Om, Ob_h2, T_func)
    return chi_pp + chi_bao + chi_h + chi_dp


def chi2_no_pp(H0, Om, Ob_h2, T_func):
    """BAO + SH0ES + DP (N=37)."""
    DM_star = DM_at_zstar(H0, Om, Ob_h2, T_func)
    rd = THETA_STAR * DM_star
    chi_bao = chi2_bao_combined(H0, Om, T_func, rd)
    chi_h = ((H0_SHOES - H0) / SIG_H0_SHOES) ** 2
    chi_dp, _, _ = chi2_DP(H0, Om, Ob_h2, T_func)
    return chi_bao + chi_h + chi_dp


def chi2_decomp(H0, Om, Ob_h2, T_func, pp=None):
    """Same as chi2_full / chi2_no_pp but returns per-channel breakdown."""
    DM_star = DM_at_zstar(H0, Om, Ob_h2, T_func)
    rd = THETA_STAR * DM_star
    chi_bao = chi2_bao_combined(H0, Om, T_func, rd)
    chi_h = ((H0_SHOES - H0) / SIG_H0_SHOES) ** 2
    chi_dp, R, lA = chi2_DP(H0, Om, Ob_h2, T_func)
    out = dict(bao=chi_bao, sh=chi_h, dp=chi_dp, rd=rd, R=R, lA=lA)
    if pp is None:
        out['pp'] = 0.0
        out['total'] = chi_bao + chi_h + chi_dp
    else:
        chi_pp, _ = chi2_panplus(H0, Om, T_func, pp)
        out['pp'] = chi_pp
        out['total'] = chi_pp + chi_bao + chi_h + chi_dp
    return out


# ============================================================================
# SECTION 8.  HISTORICAL BUG-EFFECT REPRODUCTION
# ============================================================================
#
# This section re-implements the v0.1-v1.4 BUGGY distance-prior formula and
# fits the v1.1 and v1.4 single rung against it, alongside the corrected
# paper-intent version. The bug-effect table (Tables 1, 2 of the v1.5
# paper) is produced here.
#
# After this section all subsequent fits use paper-intent only.
# ============================================================================

def _R_lA_raw_BUGGY(H0, Om, Ob_h2, T_func=None):
    """The v0.1-v1.4 buggy distance-prior integrand:
       H_E^2 = H0^2 * [ Om*(1+z)^3 + Or*(1+z)^4 + (1-Om-Or)*T^2 ]
    T is applied to the dark-energy term ONLY. Reproduces historical
    chi^2/dBIC values reported in v1.1-v1.4. NOT used outside this
    documentation section.
    """
    h = H0 / 100.0
    Om_h2 = Om * h * h
    z_star = _z_star_HS(Ob_h2, Om_h2)
    Or_frac = OMEGA_R_H2 / h**2
    if T_func is None:
        def H_E(z):
            return H0 * np.sqrt(Om*(1+z)**3 + Or_frac*(1+z)**4
                                + (1 - Om - Or_frac))
    else:
        def H_E(z):
            T_val = float(T_func(np.array([z]))[0])
            return H0 * np.sqrt(Om*(1+z)**3 + Or_frac*(1+z)**4
                                + (1 - Om - Or_frac) * T_val * T_val)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        DM_star, _ = quad(lambda z: C_KMS/H_E(z), 0, z_star, limit=400)
        rs, _ = quad(
            lambda z: 1.0/np.sqrt(3*(1 + 3*Ob_h2/(4*OMEGA_GAMMA_H2)/(1+z)))
                      * C_KMS/H_E(z),
            z_star, 1e6, limit=400)
    R_raw = np.sqrt(Om_h2) * DM_star * 100.0 / C_KMS
    lA_raw = PI * DM_star / rs
    return R_raw, lA_raw, DM_star, rs, z_star


def chi2_DP_BUGGY(H0, Om, Ob_h2, T_func=None):
    R_raw, lA_raw, _, _, _ = _R_lA_raw_BUGGY(H0, Om, Ob_h2, T_func)
    R_cal = R_raw * CAL_R
    lA_cal = lA_raw * CAL_LA
    x = np.array([R_cal, lA_cal, Ob_h2])
    xref = np.array([R_PLANCK, LA_PLANCK, WB_PLANCK])
    d = x - xref
    return float(d @ COVINV_DP @ d)


def DM_at_zstar_BUGGY(H0, Om, Ob_h2, T_func=None):
    _, _, DM_star, _, _ = _R_lA_raw_BUGGY(H0, Om, Ob_h2, T_func)
    return DM_star


def chi2_full_BUGGY(H0, Om, Ob_h2, T_func, pp):
    """v0.1-v1.4 mixed-convention chi^2:
    BAO/PP use paper-intent (H = H_LCDM * T), DP uses buggy form."""
    DM_star = DM_at_zstar_BUGGY(H0, Om, Ob_h2, T_func)
    rd = THETA_STAR * DM_star
    chi_pp, _ = chi2_panplus(H0, Om, T_func, pp)
    chi_bao = chi2_bao_combined(H0, Om, T_func, rd)
    chi_h = ((H0_SHOES - H0) / SIG_H0_SHOES) ** 2
    chi_dp = chi2_DP_BUGGY(H0, Om, Ob_h2, T_func)
    return chi_pp + chi_bao + chi_h + chi_dp


def chi2_no_pp_BUGGY(H0, Om, Ob_h2, T_func):
    DM_star = DM_at_zstar_BUGGY(H0, Om, Ob_h2, T_func)
    rd = THETA_STAR * DM_star
    chi_bao = chi2_bao_combined(H0, Om, T_func, rd)
    chi_h = ((H0_SHOES - H0) / SIG_H0_SHOES) ** 2
    chi_dp = chi2_DP_BUGGY(H0, Om, Ob_h2, T_func)
    return chi_bao + chi_h + chi_dp


def _de_nm(cost, bounds, seed=42, maxiter=60, popsize=10, nm_iter=4000,
           x0=None, skip_de=False):
    """Differential evolution + Nelder-Mead polish.
    x0 is used as init for DE if provided.
    skip_de=True: skip DE entirely and use NM polish on x0 (much faster,
    safe when x0 is known to be near the minimum, e.g. from a previously
    converged paper value).
    """
    if skip_de:
        assert x0 is not None, "skip_de requires x0"
        nm = minimize(cost, np.asarray(x0, dtype=float), method='Nelder-Mead',
                      options=dict(xatol=1e-9, fatol=1e-9, maxiter=nm_iter,
                                   adaptive=True))
        return float(nm.fun), nm.x.tolist()
    if x0 is not None:
        x0_arr = np.asarray(x0, dtype=float)
        ndim = len(x0_arr)
        pop = popsize * ndim
        rng = np.random.default_rng(seed)
        nb = max(2, pop // 2)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        x0_clip = np.clip(x0_arr, lo + 1e-6, hi - 1e-6)
        scale = 0.05 * (hi - lo)
        nbhood = x0_clip + scale * (rng.standard_normal((nb, ndim)))
        nbhood = np.clip(nbhood, lo, hi)
        rest = rng.uniform(lo, hi, size=(pop - nb, ndim))
        init = np.vstack([nbhood, rest])
        de = differential_evolution(cost, bounds, init=init, tol=1e-8,
                                    maxiter=maxiter, polish=False)
    else:
        de = differential_evolution(cost, bounds, seed=seed, tol=1e-8,
                                    maxiter=maxiter, polish=False,
                                    popsize=popsize)
    nm = minimize(cost, de.x, method='Nelder-Mead',
                  options=dict(xatol=1e-9, fatol=1e-9, maxiter=nm_iter,
                               adaptive=True))
    return float(nm.fun), nm.x.tolist()


def run_bug_section(pp=None, skip_pp_incl=False):
    """Section 8 of the paper: bug-effect tables for v1.1 and v1.4."""
    print("\n" + "=" * 76)
    print(" SECTION 8.  IMPLEMENTATION INCONSISTENCY (v1.5 paper Tables 1, 2)")
    print("=" * 76)
    print()
    print(" Two distance-prior integrands compared:")
    print("   buggy        (v0.1-v1.4): T applied to DE term only at high z")
    print("   paper-intent (this v1.5):  H_E = H_LCDM * T  uniformly")
    print()
    print(" LCDM rows are identical under both conventions (T = None).")

    # ----- PP-EXCLUDED (N=37) -----
    print("\n -- PP-excluded (N = 37) --")
    bounds_lcdm = [(60.0, 80.0), (0.20, 0.45), (0.020, 0.025)]
    bounds_lock = [(60.0, 80.0), (0.020, 0.025)]

    # LCDM
    chi_lcdm_e, x_lcdm_e = _de_nm(
        lambda x: chi2_no_pp(x[0], x[1], x[2], None),
        bounds_lcdm, x0=[70.3364, 0.27697, 0.02283])
    print(f"  LCDM (free, k=3):              chi^2 = {chi_lcdm_e:.4f}    "
          f"H0={x_lcdm_e[0]:.4f}  Om={x_lcdm_e[1]:.4f}")

    # v1.1 buggy
    chi_v11_b_e, x_v11_b_e = _de_nm(
        lambda x: chi2_no_pp_BUGGY(x[0], OM_RECURSIVE, x[1], T_v11),
        bounds_lock, x0=[69.0388, 0.02272])
    # v1.1 paper-intent
    chi_v11_f_e, x_v11_f_e = _de_nm(
        lambda x: chi2_no_pp(x[0], OM_RECURSIVE, x[1], T_v11),
        bounds_lock, x0=[68.7392, 0.02282])
    # v1.4 buggy and paper-intent
    chi_v14_b_e, _ = _de_nm(
        lambda x: chi2_no_pp_BUGGY(x[0], OM_RECURSIVE, x[1], T_v14),
        bounds_lock, x0=[69.04, 0.02272])
    chi_v14_f_e, _ = _de_nm(
        lambda x: chi2_no_pp(x[0], OM_RECURSIVE, x[1], T_v14),
        bounds_lock, x0=[68.74, 0.02282])

    N = 37; lnN = np.log(N)
    print(f"\n  {'configuration':<32}{'chi^2':>11}{'dBIC':>10}")
    print(f"  {'LCDM (k=3)':<32}{chi_lcdm_e:>11.4f}{0.0:>+10.3f}")
    for nm, chi, k in [("v1.1 buggy (k=2)",         chi_v11_b_e, 2),
                       ("v1.1 paper-intent (k=2)",  chi_v11_f_e, 2),
                       ("v1.4 buggy (k=2)",         chi_v14_b_e, 2),
                       ("v1.4 paper-intent (k=2)",  chi_v14_f_e, 2)]:
        dBIC = (chi + k*lnN) - (chi_lcdm_e + 3*lnN)
        print(f"  {nm:<32}{chi:>11.4f}{dBIC:>+10.3f}")
    print(f"\n  bug inflation on v1.1 chi^2:  "
          f"{chi_v11_f_e - chi_v11_b_e:+.4f}")
    print(f"  bug inflation on v1.4 chi^2:  "
          f"{chi_v14_f_e - chi_v14_b_e:+.4f}")

    res_exc = dict(lcdm=chi_lcdm_e,
                   v11_buggy=chi_v11_b_e, v11_fixed=chi_v11_f_e,
                   v14_buggy=chi_v14_b_e, v14_fixed=chi_v14_f_e)

    # ----- PP-INCLUDED (N=1738) -----
    if skip_pp_incl or pp is None:
        print("\n  (PP-included section skipped.)")
        return res_exc, None

    print("\n -- PP-included (N = 1738) --")

    # LCDM
    chi_lcdm_i, x_lcdm_i = _de_nm(
        lambda x: chi2_full(x[0], x[1], x[2], None, pp),
        bounds_lcdm, x0=[70.40, 0.2764, 0.02286], skip_de=True)
    print(f"  LCDM (free, k=3):              chi^2 = {chi_lcdm_i:.4f}    "
          f"H0={x_lcdm_i[0]:.4f}  Om={x_lcdm_i[1]:.4f}")

    chi_v11_b_i, _ = _de_nm(
        lambda x: chi2_full_BUGGY(x[0], OM_RECURSIVE, x[1], T_v11, pp),
        bounds_lock, x0=[69.06, 0.02275], skip_de=True)
    chi_v11_f_i, _ = _de_nm(
        lambda x: chi2_full(x[0], OM_RECURSIVE, x[1], T_v11, pp),
        bounds_lock, x0=[68.76, 0.02285], skip_de=True)
    chi_v14_b_i, _ = _de_nm(
        lambda x: chi2_full_BUGGY(x[0], OM_RECURSIVE, x[1], T_v14, pp),
        bounds_lock, x0=[69.06, 0.02275], skip_de=True)
    chi_v14_f_i, _ = _de_nm(
        lambda x: chi2_full(x[0], OM_RECURSIVE, x[1], T_v14, pp),
        bounds_lock, x0=[68.76, 0.02285], skip_de=True)

    N = 1738; lnN = np.log(N)
    print(f"\n  {'configuration':<32}{'chi^2':>12}{'dBIC':>10}")
    print(f"  {'LCDM (k=3)':<32}{chi_lcdm_i:>12.4f}{0.0:>+10.3f}")
    for nm, chi, k in [("v1.1 buggy (k=2)",         chi_v11_b_i, 2),
                       ("v1.1 paper-intent (k=2)",  chi_v11_f_i, 2),
                       ("v1.4 buggy (k=2)",         chi_v14_b_i, 2),
                       ("v1.4 paper-intent (k=2)",  chi_v14_f_i, 2)]:
        dBIC = (chi + k*lnN) - (chi_lcdm_i + 3*lnN)
        print(f"  {nm:<32}{chi:>12.4f}{dBIC:>+10.3f}")
    print(f"\n  bug inflation on v1.1 chi^2:  "
          f"{chi_v11_f_i - chi_v11_b_i:+.4f}")
    print(f"  bug inflation on v1.4 chi^2:  "
          f"{chi_v14_f_i - chi_v14_b_i:+.4f}")
    print(f"\n  Paper-reported v1.1 (buggy) PP-incl chi^2: 1601.999")
    print(f"  This script v1.1 (buggy)    PP-incl chi^2: {chi_v11_b_i:.4f}")
    print(f"  This script v1.1 (fixed)    PP-incl chi^2: {chi_v11_f_i:.4f}")

    res_inc = dict(lcdm=chi_lcdm_i,
                   v11_buggy=chi_v11_b_i, v11_fixed=chi_v11_f_i,
                   v14_buggy=chi_v14_b_i, v14_fixed=chi_v14_f_i)

    return res_exc, res_inc


# ============================================================================
# SECTION 9.  THREE-RUNG LADDER FITS  (paper Section 4)
# ============================================================================

def run_ladder_section(pp=None, skip_pp_incl=False):
    """Multi-rung ladder fits on the paper-intent pipeline:
       - free A_2 fit at n_max = 2
       - n_max scan with alternating geometric ladder, r locked to phi
       - the paper construction (3-rung lock pattern A_2 = -A_1, A_3 = -phi A_1)
    """
    print("\n" + "=" * 76)
    print(" SECTION 9.  THREE-RUNG LADDER (v1.5 paper Tables 3-7)")
    print("=" * 76)
    if pp is None or skip_pp_incl:
        print(" (PP-included; skipped if no Pantheon+ data.)")
        return None

    # Re-fit LCDM and v1.1 paper-intent baselines for reference
    chi_lcdm, x_lcdm = _de_nm(
        lambda x: chi2_full(x[0], x[1], x[2], None, pp),
        [(60,80),(0.2,0.45),(0.020,0.025)],
        x0=[70.40, 0.2764, 0.02286], skip_de=True)
    chi_v11, x_v11 = _de_nm(
        lambda x: chi2_full(x[0], OM_RECURSIVE, x[1], T_v11, pp),
        [(60,80),(0.020,0.025)], x0=[68.76, 0.02285], skip_de=True)

    N = 1738; lnN = np.log(N)

    # -----------------------------------------------------------
    # 9.1  Free A_2 (Table 3 row n=2)
    # -----------------------------------------------------------
    print("\n -- 9.1  Free-amplitude scan, n_max up to 4 (paper Table 3) --")

    def fit_nrungs(n_max, x0):
        n_free = n_max - 1  # A_2,...,A_n_max
        def cost(x):
            H0, Ob_h2 = x[0], x[1]
            amps = np.concatenate([[LNPHI_4], x[2:]])
            return chi2_full(H0, OM_RECURSIVE, Ob_h2, lambda zz: T_ladder(zz, amps), pp)
        nm = minimize(cost, np.array(x0), method='Nelder-Mead',
                      options=dict(xatol=1e-7, fatol=1e-6,
                                   maxiter=8000*(2+n_free), adaptive=True))
        return float(nm.fun), nm.x

    chi_n1, x_n1 = fit_nrungs(1, [68.76, 0.02285])
    chi_n2, x_n2 = fit_nrungs(2, [69.24, 0.02270, -0.053])
    chi_n3, x_n3 = fit_nrungs(3, list(x_n2) + [-0.05])
    chi_n4, x_n4 = fit_nrungs(4, list(x_n3) + [0.0])

    print(f"  {'n_max':>5}{'k':>3}{'chi^2':>12}{'dchi^2/n=1':>13}{'dBIC':>10}"
          f"  amplitudes")
    for nmax, chi, xfit in [(1, chi_n1, x_n1), (2, chi_n2, x_n2),
                             (3, chi_n3, x_n3), (4, chi_n4, x_n4)]:
        k = 2 if nmax == 1 else 2 + (nmax - 1)
        dBIC = (chi + k*lnN) - (chi_lcdm + 3*lnN)
        amps = ", ".join([f"A{i+2}={xfit[2+i]:+.4f}" for i in range(nmax-1)])
        print(f"  {nmax:>5d}{k:>3}{chi:>12.4f}{chi-chi_n1:>+13.4f}"
              f"{dBIC:>+10.3f}  {amps}")

    A2_best = x_n2[2]
    dchi2 = chi_n1 - chi_n2
    sigma_str = f"{np.sqrt(dchi2):.2f}" if dchi2 > 0 else "n/a"
    print(f"\n  Free A_2 best: {A2_best:+.4f}  "
          f"(v1.3 paper buggy: +0.0295)")
    print(f"  Improvement over A_2 = 0:  dchi^2 = {dchi2:+.4f}")
    print(f"  Significance away from A_2 = 0:  {sigma_str} sigma")

    # -----------------------------------------------------------
    # 9.2  n_max scan, r = phi locked alternating (Table 5)
    # -----------------------------------------------------------
    print("\n -- 9.2  Alternating geometric r=phi lock, n_max scan (Table 5) --")

    def amps_alt(r, n_max):
        return np.array([((-1)**(n+1)) * LNPHI_4 * (r**(n-1))
                         for n in range(1, n_max+1)])

    print(f"  {'n_max':>5}{'chi^2':>12}{'dBIC':>10}{'|A_nmax|':>11}")
    for n_max in range(1, 7):
        amps = amps_alt(PHI, n_max)
        def c(x): return chi2_full(x[0], OM_RECURSIVE, x[1],
                                    lambda zz: T_ladder(zz, amps), pp)
        chi, xv = _de_nm(c, [(60,80),(0.020,0.025)], x0=[69.5, 0.0225],
                          skip_de=True)
        dBIC = (chi + 2*lnN) - (chi_lcdm + 3*lnN)
        print(f"  {n_max:>5d}{chi:>12.4f}{dBIC:>+10.3f}{abs(amps[-1]):>11.5f}")

    # -----------------------------------------------------------
    # 9.3  Paper construction: three-rung lock (Table 6)
    # -----------------------------------------------------------
    print("\n -- 9.3  PAPER CONSTRUCTION (v1.5 Eq. 3 -- Table 6) --")
    print(f"\n  Locks:")
    print(f"    A_1/(ln phi)^4 = +1")
    print(f"    A_2/A_1        = -1")
    print(f"    A_3/A_1        = -phi          = {-PHI:.5f}")
    print(f"    theta_c,n      = n ln phi,  w = (ln phi)^3")
    print(f"    Omega_m        = 1/(1+e^(3 ln phi/phi))  = {OM_RECURSIVE:.6f}")

    def c_v15(x):
        return chi2_full(x[0], OM_RECURSIVE, x[1], T_v15, pp)
    chi_v15, x_v15 = _de_nm(c_v15, [(60,80),(0.020,0.025)],
                             x0=[69.91, 0.02249], skip_de=True)
    d_v15 = chi2_decomp(x_v15[0], OM_RECURSIVE, x_v15[1], T_v15, pp)

    Rs = (d_v15['R']  - R_PLANCK )/SIG_R
    lAs = (d_v15['lA'] - LA_PLANCK)/SIG_LA

    print(f"\n  Best fit: H0 = {x_v15[0]:.4f}, Omega_b h^2 = {x_v15[1]:.5f}")
    print(f"            r_d = {d_v15['rd']:.3f}  Mpc")
    print(f"  Channels: PP   = {d_v15['pp']:>8.3f}")
    print(f"            BAO  = {d_v15['bao']:>8.3f}")
    print(f"            DP   = {d_v15['dp']:>8.3f}  "
          f"(R={d_v15['R']:.5f}, lA={d_v15['lA']:.3f})")
    print(f"            SH   = {d_v15['sh']:>8.3f}")
    print(f"            TOTAL= {d_v15['total']:>8.3f}")

    dBIC_v15 = (chi_v15 + 2*lnN) - (chi_lcdm + 3*lnN)
    print(f"\n  chi^2 (k=2):     {chi_v15:.4f}")
    print(f"  dBIC vs LCDM:    {dBIC_v15:+.3f}")
    print(f"  R  - R_Planck:   {Rs:+.3f} sigma")
    print(f"  lA - lA_Planck:  {lAs:+.3f} sigma")

    # T(z) profile
    print(f"\n  T(z) profile at representative z:")
    print(f"    {'z':>8}{'theta':>10}{'T-1':>16}{'note':<25}")
    notes = {
        0.295:"DESI BGS", 0.510:"LRG1",
        1/PHI: "n=1 peak (1/phi)",
        0.706:"LRG2", 1.320:"DR2 ELG2",
        PHI: "n=2 peak (phi)",
        2.330:"Ly alpha",
        PHI**3-1: "n=3 peak",
        1090: "recombination z*",
    }
    for z in [0.0, 0.295, 0.510, 1/PHI, 0.706, 1.320, PHI, 2.330,
              PHI**3-1, 5.0, 10.0, 1090.0]:
        theta = np.log(1+z)
        Tval = float(T_v15(np.array([z]))[0])
        print(f"    {z:>8.3f}{theta:>10.4f}{Tval-1:>+16.3e}  {notes.get(z,''):<25}")

    return dict(
        lcdm=chi_lcdm, v11=chi_v11, n1=chi_n1, n2=chi_n2, n3=chi_n3,
        n4=chi_n4, v15=chi_v15, x_v15=x_v15, decomp_v15=d_v15)


# ============================================================================
# SECTION 10.  GROWTH RATE  (paper Section 5)
# ============================================================================

OR_FRAC_DEFAULT = OMEGA_R_H2 / 0.70**2

FSIG8_DATA = [
    (0.067, 0.423, 0.055, "6dFGS"),
    (0.150, 0.530, 0.085, "SDSS-MGS"),
    (0.380, 0.497, 0.045, "BOSS LOWZ"),
    (0.510, 0.458, 0.038, "BOSS CMASS"),
    (0.440, 0.413, 0.080, "WiggleZ"),
    (0.600, 0.390, 0.063, "WiggleZ"),
    (0.730, 0.437, 0.072, "WiggleZ"),
    (0.698, 0.471, 0.045, "eBOSS LRG"),
    (0.850, 0.315, 0.095, "eBOSS ELG"),
    (1.480, 0.462, 0.045, "eBOSS QSO"),
    (0.295, 0.439, 0.045, "DESI DR2 BGS"),
    (0.510, 0.476, 0.030, "DESI DR2 LRG1"),
    (0.706, 0.453, 0.022, "DESI DR2 LRG2"),
    (0.930, 0.476, 0.029, "DESI DR2 LRG3"),
    (1.317, 0.434, 0.035, "DESI DR2 ELG"),
    (1.491, 0.395, 0.045, "DESI DR2 QSO"),
]


def _lnT_and_dlnT_dN(N, amps, positions, w=LNPHI_3):
    """Compute ln T(z) and d ln T / dN at scale factor a = e^N."""
    a = np.exp(N)
    z = 1.0/a - 1.0
    theta = np.log(1.0 + z)
    T = 1.0
    dT_dN = 0.0
    for i in range(len(amps)):
        arg = (theta - positions[i]) / w
        if abs(arg) > 50:
            continue
        s2 = 1.0/np.cosh(arg)**2
        tu = np.tanh(arg)
        T = T + amps[i] * s2
        # dT/d theta = sum_i A_i * (-2/w) sech^2(arg) tanh(arg)
        # theta = -N, so dT/dN = -dT/d theta = sum A_i * (+2/w) sech^2 tanh
        dT_dN = dT_dN + amps[i] * (2.0/w) * s2 * tu
    return np.log(T), dT_dN / T


def _growth_rhs(N, y, amps, positions, alpha_g, Om, Or_frac):
    """Linear growth ODE:
       D'' + (2 + d ln H / dN) D' = (3/2) Omega_m(a) (G_eff/G) D

    Paper-intent: H = H_LCDM * T, so d ln H/dN = d ln H_LCDM/dN + d ln T/dN.
    Omega_m(a) = Om a^-3 / (E_LCDM^2 T^2)   (matter is bare, T enters via E)

    Optional G_eff modulation (v1.4 functional form, n=1 rung only):
       G_eff/G = exp[2 alpha_g sech^2((theta - ln phi)/(ln phi)^3)]
    """
    D, Dp = y
    a = np.exp(N)

    OL = 1.0 - Om - Or_frac
    H2_lcdm = Om*a**(-3) + Or_frac*a**(-4) + OL
    dH2_lcdm_dN = -3.0*Om*a**(-3) - 4.0*Or_frac*a**(-4)
    dlnH_lcdm = 0.5 * dH2_lcdm_dN / H2_lcdm

    lnT, dlnT = _lnT_and_dlnT_dN(N, amps, positions)
    T = np.exp(lnT)
    dlnH = dlnH_lcdm + dlnT

    Omega_m_a = Om * a**(-3) / (H2_lcdm * T*T)

    if alpha_g == 0.0:
        Geff = 1.0
    else:
        z_val = 1.0/a - 1.0
        theta_val = np.log(1.0 + z_val)
        arg = np.clip((theta_val - THETA_C_PHI)/LNPHI_3, -50.0, 50.0)
        s2 = 1.0/np.cosh(arg)**2
        Geff = np.exp(2.0 * alpha_g * s2)

    Dpp = -(2.0 + dlnH)*Dp + 1.5*Omega_m_a*Geff*D
    return [Dp, Dpp]


def solve_growth(amps, positions, alpha_g, Om=OM_RECURSIVE,
                 Or_frac=OR_FRAC_DEFAULT,
                 N_init=-6.0, N_end=0.0, n_points=2000):
    a_init = np.exp(N_init)
    N_eval = np.linspace(N_init, N_end, n_points)
    sol = solve_ivp(_growth_rhs, [N_init, N_end], [a_init, a_init],
                    args=(amps, positions, alpha_g, Om, Or_frac),
                    method='DOP853', t_eval=N_eval,
                    rtol=1e-10, atol=1e-12)
    return sol


def fsigma8_predict(z_array, amps, positions, alpha_g, Om=OM_RECURSIVE):
    """Predict f sigma_8(z) for the given ladder + optional G_eff coupling.
    sigma_8(z=0) is set relative to a LCDM reference at the same Om."""
    sol = solve_growth(amps, positions, alpha_g, Om=Om)
    D_arr = sol.y[0]; Dp_arr = sol.y[1]; f_arr = Dp_arr / D_arr
    D0 = D_arr[-1]
    sol_ref = solve_growth(np.array([]), np.array([]), 0.0, Om=Om)
    D0_ref = sol_ref.y[0, -1]
    sigma8_0 = SIGMA8_PLANCK * D0 / D0_ref

    z_arr = np.asarray(z_array, dtype=float)
    N_targets = -np.log(1.0 + z_arr)
    D_z = np.interp(N_targets, sol.t, D_arr)
    f_z = np.interp(N_targets, sol.t, f_arr)
    return f_z * (D_z / D0) * sigma8_0, sigma8_0


def chi2_fsigma8(amps, positions, alpha_g, Om=OM_RECURSIVE):
    z_vals = np.array([d[0] for d in FSIG8_DATA])
    fs_obs = np.array([d[1] for d in FSIG8_DATA])
    fs_err = np.array([d[2] for d in FSIG8_DATA])
    fs_pred, s8 = fsigma8_predict(z_vals, amps, positions, alpha_g, Om)
    return float(np.sum((fs_obs - fs_pred)**2 / fs_err**2)), s8


def run_fsigma8_section():
    """Section 10 of the paper:
       - alpha_g = 0 chi^2 for LCDM / v1.1 / 3-rung
       - alpha_g scan at the 3-rung locks, including alpha_g = -5 (ln phi)^4
    """
    print("\n" + "=" * 76)
    print(" SECTION 10.  f sigma_8 FIT  (v1.5 paper Tables 8, 9)")
    print("=" * 76)

    # alpha_g = 0 table
    print(f"\n -- 10.1  alpha_g = 0 comparison (Table 8) --\n")
    chi_l, s8_l = chi2_fsigma8(np.array([]), np.array([]), 0.0)
    chi_v11, s8_v11 = chi2_fsigma8(np.array([LNPHI_4]),
                                    np.array([THETA_C_PHI]), 0.0)
    chi_v15, s8_v15 = chi2_fsigma8(LADDER_V15_AMPS, LADDER_V15_POS, 0.0)
    print(f"  {'model':<35}{'chi^2':>10}{'sigma_8(0)':>13}")
    print(f"  {'LCDM (Om-lock)':<35}{chi_l:>10.3f}{s8_l:>13.5f}")
    print(f"  {'v1.1 single rung (paper-intent)':<35}{chi_v11:>10.3f}{s8_v11:>13.5f}")
    print(f"  {'3-rung v1.5 construction':<35}{chi_v15:>10.3f}{s8_v15:>13.5f}")

    # alpha_g scan (Table 9)
    print(f"\n -- 10.2  alpha_g scan at 3-rung locks (Table 9) --\n")
    print(f"  {'lock':<22}{'alpha_g':>11}{'chi^2_fs':>10}{'sigma_8':>10}")
    for name, ag in [
        ("0",                  0.0),
        ("-A_1 = -(lnphi)^4",  -LNPHI_4),
        ("-2 A_1",             -2*LNPHI_4),
        ("-3 A_1",             -3*LNPHI_4),
        ("-4 A_1",             -4*LNPHI_4),
        ("-5 A_1",             -5*LNPHI_4),
        ("-8 A_1",             -8*LNPHI_4),
    ]:
        chi_fs, s8 = chi2_fsigma8(LADDER_V15_AMPS, LADDER_V15_POS, ag)
        print(f"  {name:<22}{ag:>+11.5f}{chi_fs:>10.3f}{s8:>10.5f}")

    print(f"\n  Paper-construction lock:  alpha_g = -5 (ln phi)^4 = "
          f"{-5*LNPHI_4:+.5f}")
    chi_p, s8_p = chi2_fsigma8(LADDER_V15_AMPS, LADDER_V15_POS, -5*LNPHI_4)
    print(f"      chi^2_fs = {chi_p:.3f},  sigma_8(0) = {s8_p:.5f}")
    print(f"      sigma_8 - sigma_8_Planck = {s8_p - SIGMA8_PLANCK:+.5f}  "
          f"({(s8_p-SIGMA8_PLANCK)/0.006:+.3f} sigma)")

    return dict(lcdm=chi_l, v11=chi_v11, v15_ag0=chi_v15,
                v15_paper=chi_p, s8_lcdm=s8_l, s8_v11=s8_v11,
                s8_v15_ag0=s8_v15, s8_v15_paper=s8_p)


# ============================================================================
# SECTION 11.  JOINT BACKGROUND + fsigma8  (paper Table 10)
# ============================================================================

def run_joint_section(pp=None, skip_pp_incl=False):
    """Joint chi^2 = background chi^2 + fsigma8 chi^2 at the paper locks."""
    print("\n" + "=" * 76)
    print(" SECTION 11.  JOINT BACKGROUND + fsigma8  (v1.5 paper Table 10)")
    print("=" * 76)

    # Paper construction with alpha_g = -5 (ln phi)^4
    alpha_g_paper = -5 * LNPHI_4

    # ----- PP-included -----
    chi_fs_v15, s8_v15 = chi2_fsigma8(LADDER_V15_AMPS, LADDER_V15_POS,
                                       alpha_g_paper)
    chi_fs_lcdm, s8_lcdm = chi2_fsigma8(np.array([]), np.array([]), 0.0)

    if pp is not None and not skip_pp_incl:
        # LCDM
        chi_bg_lcdm, x_lcdm = _de_nm(
            lambda x: chi2_full(x[0], x[1], x[2], None, pp),
            [(60,80),(0.2,0.45),(0.020,0.025)],
            x0=[70.40, 0.2764, 0.02286], skip_de=True)
        # v1.5
        chi_bg_v15, x_v15 = _de_nm(
            lambda x: chi2_full(x[0], OM_RECURSIVE, x[1], T_v15, pp),
            [(60,80),(0.020,0.025)], x0=[69.91, 0.02249], skip_de=True)

        chi_j_lcdm = chi_bg_lcdm + chi_fs_lcdm
        chi_j_v15  = chi_bg_v15  + chi_fs_v15

        Nj = 1738 + 16
        lnNj = np.log(Nj)
        dBIC = (chi_j_v15 + 2*lnNj) - (chi_j_lcdm + 3*lnNj)

        print(f"\n -- PP-included (N_joint = {Nj}) --")
        print(f"  {'model':<35}{'k':>3}{'chi_bg':>12}{'chi_fs8':>10}"
              f"{'joint':>12}{'dBIC':>10}")
        print(f"  {'LCDM':<35}{3:>3}{chi_bg_lcdm:>12.4f}"
              f"{chi_fs_lcdm:>10.4f}{chi_j_lcdm:>12.4f}{0.0:>+10.3f}")
        print(f"  {'3-rung (alpha_g = -5 (lnphi)^4)':<35}{2:>3}"
              f"{chi_bg_v15:>12.4f}{chi_fs_v15:>10.4f}{chi_j_v15:>12.4f}"
              f"{dBIC:>+10.3f}")
        print(f"\n  Best fit (3-rung): H0 = {x_v15[0]:.4f},  "
              f"Omega_b h^2 = {x_v15[1]:.5f}")
        print(f"  sigma_8(0) = {s8_v15:.5f}  "
              f"(Planck: 0.811 +/- 0.006; dev = "
              f"{(s8_v15-SIGMA8_PLANCK)/0.006:+.3f} sigma)")

    # ----- PP-excluded -----
    chi_bg_lcdm_e, x_lcdm_e = _de_nm(
        lambda x: chi2_no_pp(x[0], x[1], x[2], None),
        [(60,80),(0.2,0.45),(0.020,0.025)],
        x0=[70.3364, 0.27697, 0.02283])
    chi_bg_v15_e, x_v15_e = _de_nm(
        lambda x: chi2_no_pp(x[0], OM_RECURSIVE, x[1], T_v15),
        [(60,80),(0.020,0.025)], x0=[69.89, 0.02246])

    chi_j_lcdm_e = chi_bg_lcdm_e + chi_fs_lcdm
    chi_j_v15_e  = chi_bg_v15_e  + chi_fs_v15

    Nj_e = 37 + 16
    lnNj_e = np.log(Nj_e)
    dBIC_e = (chi_j_v15_e + 2*lnNj_e) - (chi_j_lcdm_e + 3*lnNj_e)

    print(f"\n -- PP-excluded (N_joint = {Nj_e}) --")
    print(f"  {'model':<35}{'k':>3}{'chi_bg':>12}{'chi_fs8':>10}"
          f"{'joint':>12}{'dBIC':>10}")
    print(f"  {'LCDM':<35}{3:>3}{chi_bg_lcdm_e:>12.4f}"
          f"{chi_fs_lcdm:>10.4f}{chi_j_lcdm_e:>12.4f}{0.0:>+10.3f}")
    print(f"  {'3-rung (alpha_g = -5 (lnphi)^4)':<35}{2:>3}"
          f"{chi_bg_v15_e:>12.4f}{chi_fs_v15:>10.4f}{chi_j_v15_e:>12.4f}"
          f"{dBIC_e:>+10.3f}")
    print(f"\n  Best fit (3-rung): H0 = {x_v15_e[0]:.4f},  "
          f"Omega_b h^2 = {x_v15_e[1]:.5f}")
    print(f"\n  H0 stability (PP-inc - PP-exc): "
          f"{x_v15[0] - x_v15_e[0]:+.4f}  km/s/Mpc" if pp is not None and not skip_pp_incl else "")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LVC v1.5 reproduction script "
                    "(paper-intent everywhere; v0.1-v1.4 bug corrected).")
    parser.add_argument('--skip-pp-included', action='store_true',
                        help="Skip Pantheon+ sections (PP-excluded only).")
    parser.add_argument('--skip-fsigma8', action='store_true',
                        help="Skip fsigma8 section (Section 10).")
    parser.add_argument('--skip-bug-section', action='store_true',
                        help="Skip historical bug-effect table (Section 8).")
    parser.add_argument('--skip-ladder', action='store_true',
                        help="Skip ladder fits (Section 9).")
    parser.add_argument('--skip-joint', action='store_true',
                        help="Skip joint bg+fs8 fit (Section 11).")
    args = parser.parse_args()

    print("=" * 76)
    print(" LVC final-v1.5 reproduction script")
    print(" Paper-intent likelihood (uniform multiplicative T); v0.1-v1.4 bug corrected.")
    print("=" * 76)
    print()
    print(f"  Constants:")
    print(f"    phi             = {PHI:.10f}")
    print(f"    ln phi          = {THETA_C_PHI:.10f}")
    print(f"    (ln phi)^3 = w  = {LNPHI_3:.10f}")
    print(f"    (ln phi)^4 = A_1= {LNPHI_4:.10f}")
    print(f"    Omega_m lock    = {OM_RECURSIVE:.10f}")
    print(f"    sigma_8 Planck  = {SIGMA8_PLANCK:.4f}")

    # Load Pantheon+ if needed
    pp = None
    if not args.skip_pp_included:
        print(f"\n  Loading Pantheon+ data from {PP_DIR} ...")
        pp = load_pantheonplus()
        print(f"  Loaded.  N_SN = {pp['N']}")

    t0 = time.time()

    if not args.skip_bug_section:
        run_bug_section(pp=pp, skip_pp_incl=args.skip_pp_included)

    if not args.skip_ladder:
        run_ladder_section(pp=pp, skip_pp_incl=args.skip_pp_included)

    if not args.skip_fsigma8:
        run_fsigma8_section()

    if not args.skip_joint:
        run_joint_section(pp=pp, skip_pp_incl=args.skip_pp_included)

    print("\n" + "=" * 76)
    print(f" Reproduction complete in {time.time()-t0:.1f} s")
    print("=" * 76)


if __name__ == "__main__":
    main()
