# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LUMENPIXEL
#
# Released under the MIT License.

"""
LVC final-v1.3 -- Single-file reproduction script
==================================================

Reproduces the numerical results in the final-v1.3 working paper:

    "Lagrangian Variable Cosmology, final-v1.3:
     Phenomenological Exploration of a Multi-Rung Generalisation
     of the Localised Modulation in Late-Time Expansion History"
                                LUMENPIXEL, May 2026

This is a single self-contained Python file. The required external
dependencies are numpy, pandas, scipy. CAMB is optional and is used
only by the sigma_8 cross-check section (Table 3, right column); the
script gracefully skips that section if CAMB is not installed.

The likelihood pipeline (Sections 1-7 below) is identical to v1.1,
which is reproduced as part of the sanity-check phase.

Reproduces (key paper numbers, PP-excluded N=37 unless noted):
---------------------------------------------------------------
Section 8  Sanity-check baselines
    Lambda-CDM (k=3)                    chi^2 = 78.962
    v1.1 single rung (k=2)              chi^2 = 56.134     dBIC = -26.44
    PP-included v1.1 (k=2, N=1738)      chi^2 = 1601.999   dBIC = -30.31

Section 9  Multi-rung family fits (Table 1)
    Alt sign, A_n = (-1)^(n+1) phi^(n-1) A_1, n_max=2     chi^2 = 92.322
                                                          dBIC = +9.75
    Same sign, A_n = phi^-(n-1) A_1, n_max=2              chi^2 = 53.719
                                                          dBIC = -28.85
    Same sign, A_n = phi^-(n-1) A_1, n_max=3              chi^2 = 53.282
                                                          dBIC = -29.29

Section 10  A_2 profile likelihood (Table 2; PP-included N=1738)
    A_2 free fit                  A_2 = +0.0295   chi^2 = 1599.805
    A_2 = +(ln phi)^4 / phi       A_2 = +0.0331   chi^2 = 1599.837
    A_2 = -(ln phi)^4 * phi       A_2 = -0.0868   chi^2 = 1637.747
    A_2 = 0 (v1.1 single rung)                    chi^2 = 1601.999

Section 11  sigma_8 (Table 3; direct growth ODE)
    LCDM at Om = Om_lock = 0.2907           sigma_8 = 0.787
    v1.1 single rung                        sigma_8 = 0.788
    Same-sign damped ladder n_max=2         sigma_8 = 0.782
    Same-sign damped ladder n_max=3         sigma_8 = 0.778
    Alt-sign amplified ladder n_max=2       sigma_8 = 0.806
    Planck-LCDM reference                   sigma_8 = 0.811

Section 12  CAMB cross-check (Table 3 right column; OPTIONAL)

Section 13  Non-minimal coupling alpha scan (Table 4)
    With f(vartheta) = exp(-alpha (1-cos(vartheta/2))) :
        v1.1 single + alpha=+0.2258  -->  sigma_8 = 0.811
        Same-sign damped n=2 + alpha=+0.1176  -->  sigma_8 = 0.811
        Same-sign damped n=3 + alpha=+0.0796  -->  sigma_8 = 0.811

External data
-------------
Pantheon+ data (~33 MB) and covariance must be available locally for the
PP-included (N=1738) sections:
    Pantheon+SH0ES.dat
    Pantheon+SH0ES_STAT+SYS.cov

If absent, the script offers automatic git-based download. PP-excluded
sections (N=37) do not require these files.

Usage
-----
    python lvc_v13_reproduce.py                  # default (full run)
    python lvc_v13_reproduce.py --skip-pp-included   # PP-excluded only
    python lvc_v13_reproduce.py --skip-camb      # no CAMB cross-check
    python lvc_v13_reproduce.py --help           # full options

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
from scipy.interpolate import interp1d
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
INV_PHI    = 1.0 / PHI
THETA_C_PHI = np.log(PHI)
LNPHI_3 = THETA_C_PHI ** 3
LNPHI_4 = THETA_C_PHI ** 4

OM_RECURSIVE = 1.0 / (1.0 + np.exp(3.0 * THETA_C_PHI / PHI))   # 0.290653

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

N_PP_EXCLUDED = 37
N_PP_INCLUDED = 1701 + 13 + 11 + 2 + 7 + 3 + 1   # = 1738


# ============================================================================
# SECTION 2.  PANTHEON+ DATA LOADER (with auto-download)
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
    return np.sqrt(Om*(1+z)**3 + (1-Om))


def T_ladder(z, n_max=1, A_factor=1.0, sign='same'):
    """
    Multi-rung sech^2 ladder modulation in theta = ln(1+z):

      T(z) = 1 + sum_{n=1}^{n_max} sign_n * A_factor^(n-1) * (ln phi)^4 *
                  sech^2((theta - n ln phi) / (ln phi)^3)

    sign='alt' uses sign_n = (-1)^(n+1)  (alternating)
    sign='same' uses sign_n = +1 always  (uniform)
    """
    z = np.asarray(z, dtype=float)
    theta = np.log(1.0 + z)
    bump = np.zeros_like(theta)
    for n in range(1, n_max + 1):
        sign_n = (1.0 if n % 2 == 1 else -1.0) if sign == 'alt' else 1.0
        amp = sign_n * (A_factor ** (n - 1)) * LNPHI_4
        arg = np.clip((theta - n * THETA_C_PHI) / LNPHI_3, -50.0, 50.0)
        bump = bump + amp / np.cosh(arg) ** 2
    return 1.0 + bump


def T_two_rung_freeA2(z, A2):
    """v1.1 first rung locked, second rung amplitude free."""
    z = np.asarray(z, dtype=float); theta = np.log(1+z)
    a1 = np.clip((theta - THETA_C_PHI)/LNPHI_3, -50, 50)
    a2 = np.clip((theta - 2*THETA_C_PHI)/LNPHI_3, -50, 50)
    return 1 + LNPHI_4 / np.cosh(a1)**2 + A2 / np.cosh(a2)**2


def comoving_dist_grid(H0, Om, T_func=None):
    if T_func is None:
        H = H0 * E_lcdm(_Z_GRID, Om)
    else:
        Tv = T_func(_Z_GRID)
        if np.min(Tv) < 0.4 or np.max(Tv) > 2.0:
            return None
        H = H0 * E_lcdm(_Z_GRID, Om) * Tv
    if np.any(H <= 0):
        return None
    integrand = C_KMS / H
    DC = np.zeros_like(_Z_GRID)
    DC[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1])
                       * np.diff(_Z_GRID))
    return DC


# ============================================================================
# SECTION 4.  PANTHEON+ LIKELIHOOD
# ============================================================================

def chi2_panplus(H0, Om, T_func, pp):
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
# SECTION 5.  BAO DATA TABLES
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
# SECTION 6.  DISTANCE PRIORS
# ============================================================================

def _z_star_HS(Ob_h2, Om_h2):
    g1 = 0.0783 * Ob_h2**(-0.238) / (1 + 39.5 * Ob_h2**0.763)
    g2 = 0.560 / (1 + 21.1 * Ob_h2**1.81)
    return 1048 * (1 + 0.00124 * Ob_h2**(-0.738)) * (1 + g1 * Om_h2**g2)


def _R_lA_raw(H0, Om, Ob_h2, T_func=None):
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


_R_AT_PLANCK, _LA_AT_PLANCK, _, _, _ = _R_lA_raw(67.36, 0.3153, 0.02237, None)
CAL_R = R_PLANCK / _R_AT_PLANCK
CAL_LA = LA_PLANCK / _LA_AT_PLANCK


def chi2_DP(H0, Om, Ob_h2, T_func=None):
    R_raw, lA_raw, _, _, _ = _R_lA_raw(H0, Om, Ob_h2, T_func)
    R_cal = R_raw * CAL_R
    lA_cal = lA_raw * CAL_LA
    x = np.array([R_cal, lA_cal, Ob_h2])
    xref = np.array([R_PLANCK, LA_PLANCK, WB_PLANCK])
    d = x - xref
    return float(d @ COVINV_DP @ d), float(R_cal), float(lA_cal)


def DM_at_zstar(H0, Om, Ob_h2, T_func=None):
    _, _, DM_star, _, _ = _R_lA_raw(H0, Om, Ob_h2, T_func)
    return DM_star


# ============================================================================
# SECTION 7.  COMBINED LIKELIHOODS
# ============================================================================

def chi2_bao_combined(H0, Om, T_func, rd):
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
    """BAO + SH0ES + DP only (N=37)."""
    DM_star = DM_at_zstar(H0, Om, Ob_h2, T_func)
    rd = THETA_STAR * DM_star
    chi_bao = chi2_bao_combined(H0, Om, T_func, rd)
    chi_h = ((H0_SHOES - H0) / SIG_H0_SHOES) ** 2
    chi_dp, _, _ = chi2_DP(H0, Om, Ob_h2, T_func)
    return chi_bao + chi_h + chi_dp


# ============================================================================
# SECTION 8.  SANITY-CHECK BASELINES
# ============================================================================

def _de_nm(cost, bounds, seed=42, maxiter=50, popsize=10, nm_iter=3000):
    de = differential_evolution(cost, bounds, seed=seed, tol=1e-7,
                                maxiter=maxiter, polish=False, popsize=popsize)
    nm = minimize(cost, de.x, method='Nelder-Mead',
                  options=dict(xatol=1e-8, fatol=1e-8, maxiter=nm_iter,
                               adaptive=True))
    return float(nm.fun), nm.x.tolist()


def _nm_only(cost, x0, nm_iter=2500):
    res = minimize(cost, x0, method='Nelder-Mead',
                   options=dict(xatol=1e-8, fatol=1e-8, maxiter=nm_iter,
                                adaptive=True))
    return float(res.fun), res.x.tolist()


def run_baselines(pp=None):
    print("=" * 76)
    print("Sec. 8 BASELINES")
    print("=" * 76)

    # PP-excluded LCDM
    def cost_lcdm(p):
        H0, Om, Ob = p
        if not (60<=H0<=80 and 0.18<=Om<=0.45 and 0.018<=Ob<=0.028): return 1e10
        c = chi2_no_pp(H0, Om, Ob, None)
        return c if np.isfinite(c) else 1e10

    print("\n[A] LCDM PP-excluded (k=3) ...")
    t0 = time.time()
    chi_lcdm, x_lcdm = _de_nm(cost_lcdm,
        [(60,80),(0.20,0.40),(0.020,0.026)], maxiter=80, popsize=12)
    print(f"  chi^2 = {chi_lcdm:.4f}  H0={x_lcdm[0]:.3f}  Om={x_lcdm[1]:.4f}  "
          f"Ob={x_lcdm[2]:.5f}  [{time.time()-t0:.1f}s]")

    # PP-excluded v1.1 single rung
    def cost_v11(p):
        H0, Ob = p
        if not (60<=H0<=80 and 0.018<=Ob<=0.028): return 1e10
        T = lambda z: T_ladder(z, n_max=1)
        c = chi2_no_pp(H0, OM_RECURSIVE, Ob, T)
        return c if np.isfinite(c) else 1e10

    print("\n[B] v1.1 single rung PP-excluded (k=2) ...")
    t0 = time.time()
    chi_v11, x_v11 = _de_nm(cost_v11, [(60,80),(0.020,0.026)],
                            maxiter=50, popsize=10)
    bic_lcdm = chi_lcdm + 3*np.log(N_PP_EXCLUDED)
    bic_v11 = chi_v11 + 2*np.log(N_PP_EXCLUDED)
    print(f"  chi^2 = {chi_v11:.4f}  H0={x_v11[0]:.3f}  Ob={x_v11[1]:.5f}")
    print(f"  dBIC vs LCDM = {bic_v11-bic_lcdm:+.3f}  [{time.time()-t0:.1f}s]")
    print(f"  (paper value: dBIC = -26.44)")

    if pp is None:
        return dict(chi_lcdm=chi_lcdm, chi_v11=chi_v11)

    # PP-included LCDM
    def cost_lcdm_pp(p):
        H0, Om, Ob = p
        if not (60<=H0<=80 and 0.18<=Om<=0.45 and 0.018<=Ob<=0.028): return 1e10
        c = chi2_full(H0, Om, Ob, None, pp)
        return c if np.isfinite(c) else 1e10

    print("\n[C] LCDM PP-included (k=3) ...")
    t0 = time.time()
    chi_lcdm_pp, x_lcdm_pp = _de_nm(cost_lcdm_pp,
        [(60,80),(0.20,0.40),(0.020,0.026)], maxiter=60, popsize=10)
    print(f"  chi^2 = {chi_lcdm_pp:.4f}  H0={x_lcdm_pp[0]:.3f}  "
          f"Om={x_lcdm_pp[1]:.4f}  [{time.time()-t0:.1f}s]")

    # PP-included v1.1
    def cost_v11_pp(p):
        H0, Ob = p
        if not (60<=H0<=80 and 0.018<=Ob<=0.028): return 1e10
        T = lambda z: T_ladder(z, n_max=1)
        c = chi2_full(H0, OM_RECURSIVE, Ob, T, pp)
        return c if np.isfinite(c) else 1e10

    print("\n[D] v1.1 single rung PP-included (k=2) ...")
    t0 = time.time()
    chi_v11_pp, x_v11_pp = _de_nm(cost_v11_pp,
        [(60,80),(0.020,0.026)], maxiter=40, popsize=10)
    bic_lcdm_pp = chi_lcdm_pp + 3*np.log(N_PP_INCLUDED)
    bic_v11_pp = chi_v11_pp + 2*np.log(N_PP_INCLUDED)
    print(f"  chi^2 = {chi_v11_pp:.4f}  H0={x_v11_pp[0]:.3f}  Ob={x_v11_pp[1]:.5f}")
    print(f"  dBIC vs LCDM = {bic_v11_pp-bic_lcdm_pp:+.3f}  [{time.time()-t0:.1f}s]")
    print(f"  (paper value: dBIC = -30.31)")

    return dict(chi_lcdm=chi_lcdm, chi_v11=chi_v11,
                chi_lcdm_pp=chi_lcdm_pp, chi_v11_pp=chi_v11_pp,
                H0_v11=x_v11[0], H0_v11_pp=x_v11_pp[0])


# ============================================================================
# SECTION 9.  MULTI-RUNG FAMILY FITS (Table 1)
# ============================================================================

def run_family_fits(pp=None):
    print("\n" + "=" * 76)
    print("Sec. 9 MULTI-RUNG FAMILY FITS (Table 1)")
    print("=" * 76)

    families = [
        ("Alt sign, A_factor=phi, n_max=2",   2, PHI,   'alt'),
        ("Same sign, A_factor=1/phi, n_max=2", 2, 1/PHI,'same'),
        ("Same sign, A_factor=1/phi, n_max=3", 3, 1/PHI,'same'),
    ]

    results = {}
    for label, nm_v, Af, sign in families:
        def cost(p):
            H0, Ob = p
            if not (60<=H0<=80 and 0.018<=Ob<=0.028): return 1e10
            T = lambda z: T_ladder(z, n_max=nm_v, A_factor=Af, sign=sign)
            return chi2_no_pp(H0, OM_RECURSIVE, Ob, T)
        t0 = time.time()
        chi, x = _de_nm(cost, [(60,80),(0.020,0.026)],
                         maxiter=40, popsize=10)
        print(f"\n  {label}:")
        print(f"    PP-excluded chi^2 = {chi:.4f}  H0={x[0]:.3f}  Ob={x[1]:.5f}  [{time.time()-t0:.1f}s]")
        results[label] = dict(chi_pp_excl=chi, H0_pp_excl=x[0])

        if pp is not None:
            def cost_pp(p):
                H0, Ob = p
                if not (60<=H0<=80 and 0.018<=Ob<=0.028): return 1e10
                T = lambda z: T_ladder(z, n_max=nm_v, A_factor=Af, sign=sign)
                return chi2_full(H0, OM_RECURSIVE, Ob, T, pp)
            t0 = time.time()
            chi_pp, x_pp = _de_nm(cost_pp, [(60,80),(0.020,0.026)],
                                   maxiter=40, popsize=10)
            print(f"    PP-included chi^2 = {chi_pp:.4f}  H0={x_pp[0]:.3f}  [{time.time()-t0:.1f}s]")
            results[label]['chi_pp_incl'] = chi_pp
            results[label]['H0_pp_incl'] = x_pp[0]

    return results


# ============================================================================
# SECTION 10.  A_2 PROFILE LIKELIHOOD (Table 2)
# ============================================================================

def run_A2_profile(pp):
    print("\n" + "=" * 76)
    print("Sec. 10 A_2 PROFILE LIKELIHOOD (Table 2; PP-included, N=1738)")
    print("=" * 76)

    if pp is None:
        print("  [SKIPPED: PP data not loaded]")
        return None

    # Free A_2 best
    def cost_free(p):
        H0, Ob, A2 = p
        if not (60<=H0<=80 and 0.018<=Ob<=0.028 and -0.10<=A2<=0.10):
            return 1e10
        T = lambda z: T_two_rung_freeA2(z, A2)
        return chi2_full(H0, OM_RECURSIVE, Ob, T, pp)

    print("\n[1] Free A_2 (k=3) ...")
    t0 = time.time()
    best = (1e10, None)
    for x0 in [[69.06, 0.02275, 0.029], [69.06, 0.02275, 0.0],
               [69.06, 0.02275, 0.04]]:
        chi, x = _nm_only(cost_free, x0, nm_iter=2000)
        if chi < best[0]: best = (chi, x)
    chi_free, x_free = best
    print(f"  chi^2 = {chi_free:.4f}  A_2 = {x_free[2]:+.5f}  H0={x_free[0]:.3f}  [{time.time()-t0:.1f}s]")

    # Lock points
    def cost_at(A2_fixed):
        def c(p):
            H0, Ob = p
            if not (60<=H0<=80 and 0.018<=Ob<=0.028): return 1e10
            T = lambda z: T_two_rung_freeA2(z, A2_fixed)
            return chi2_full(H0, OM_RECURSIVE, Ob, T, pp)
        return _nm_only(c, [69.06, 0.02275], nm_iter=1500)

    locks = [
        (0.0, 'A_2 = 0 (v1.1 single rung)'),
        (LNPHI_4/PHI, 'A_2 = +(ln phi)^4 / phi  (damping lock)'),
        (-LNPHI_4*PHI, 'A_2 = -(ln phi)^4 * phi  (amplification lock)'),
    ]
    table_rows = []
    for A2v, label in locks:
        t0 = time.time()
        chi, x = cost_at(A2v)
        table_rows.append((label, A2v, chi, chi - chi_free))
        print(f"\n[{label}]")
        print(f"  chi^2 = {chi:.4f}   dchi^2 vs free = {chi-chi_free:+.4f}   [{time.time()-t0:.1f}s]")

    # Coarse profile for 1-sigma interval
    print("\n[3] Coarse profile chi^2(A_2):")
    print(f"  {'A_2':>9s}  {'chi^2':>10s}  {'dchi^2':>9s}")
    A2_grid = np.array([-0.04, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015,
                        0.02, 0.025, 0.029, 0.033, 0.04, 0.05, 0.06, 0.08, 0.10])
    chis_grid = []
    for A2v in A2_grid:
        chi, _ = cost_at(float(A2v))
        chis_grid.append(chi)
        print(f"  {A2v:>+9.5f}  {chi:>10.4f}  {chi-chi_free:>+9.4f}")
    chis_arr = np.array(chis_grid)
    dchi = chis_arr - chi_free
    mask1 = dchi < 1.0
    if mask1.any():
        print(f"\n  1-sigma interval (chi^2 < free+1): "
              f"[{A2_grid[mask1].min():+.5f}, {A2_grid[mask1].max():+.5f}]")
    mask4 = dchi < 4.0
    if mask4.any():
        print(f"  2-sigma interval (chi^2 < free+4): "
              f"[{A2_grid[mask4].min():+.5f}, {A2_grid[mask4].max():+.5f}]")
    A2_0_dchi = float(np.interp(0.0, A2_grid, dchi))
    print(f"\n  A_2 = 0 sits at {np.sqrt(max(A2_0_dchi,0)):.2f} sigma from free best")
    return dict(A2_free=x_free[2], chi_free=chi_free,
                rows=table_rows)


# ============================================================================
# SECTION 11.  sigma_8 VIA DIRECT GROWTH ODE (Table 3, left column)
# ============================================================================

def solve_growth(T_func, Om, Or=0.0, a_init=1e-3, a_end=1.0, n_pts=400):
    """Linear growth ODE in flat FLRW with H(z) = H_LCDM(z) * T(z)."""
    lna_grid = np.linspace(np.log(a_init), np.log(a_end), n_pts)
    a_grid = np.exp(lna_grid)
    z_grid = 1.0/a_grid - 1.0
    Olam = 1.0 - Om - Or
    E_grid = np.sqrt(Om*(1+z_grid)**3 + Or*(1+z_grid)**4 + Olam)
    if T_func is not None:
        E_grid = E_grid * T_func(z_grid)
    lnE = np.log(E_grid)
    dlnE_dlna = np.gradient(lnE, lna_grid)
    Om_a = Om * a_grid**(-3) / E_grid**2
    dlnE_int = interp1d(lna_grid, dlnE_dlna, kind='cubic',
                        fill_value='extrapolate')
    Om_a_int = interp1d(lna_grid, Om_a, kind='cubic',
                        fill_value='extrapolate')

    def rhs(lna, y):
        D, dD = y
        return [dD, -(2.0 + dlnE_int(lna))*dD + 1.5*Om_a_int(lna)*D]

    sol = solve_ivp(rhs, [lna_grid[0], lna_grid[-1]],
                    [a_init, a_init],
                    t_eval=lna_grid, method='Radau',
                    rtol=1e-9, atol=1e-11)
    return a_grid, sol.y[0]


def sigma_8(T_func, Om, sigma8_LCDM_ref=0.811, Om_LCDM_ref=0.3153):
    a_m, D_m = solve_growth(T_func, Om)
    a_p, D_p = solve_growth(None, Om_LCDM_ref)
    return sigma8_LCDM_ref * D_m[-1] / D_p[-1]


def solve_growth_Geff(T_func, G_func, Om, Or=0.0,
                      a_init=1e-3, a_end=1.0, n_pts=400):
    """Growth ODE with effective Newton constant G_eff(z)."""
    lna_grid = np.linspace(np.log(a_init), np.log(a_end), n_pts)
    a_grid = np.exp(lna_grid)
    z_grid = 1.0/a_grid - 1.0
    Olam = 1.0 - Om - Or
    E_grid = np.sqrt(Om*(1+z_grid)**3 + Or*(1+z_grid)**4 + Olam)
    if T_func is not None:
        E_grid = E_grid * T_func(z_grid)
    lnE = np.log(E_grid)
    dlnE_dlna = np.gradient(lnE, lna_grid)
    Om_a = Om * a_grid**(-3) / E_grid**2
    G_grid = G_func(z_grid) if G_func is not None else np.ones_like(z_grid)
    dlnE_int = interp1d(lna_grid, dlnE_dlna, kind='cubic', fill_value='extrapolate')
    Om_a_int = interp1d(lna_grid, Om_a, kind='cubic', fill_value='extrapolate')
    G_int    = interp1d(lna_grid, G_grid, kind='cubic', fill_value='extrapolate')

    def rhs(lna, y):
        D, dD = y
        return [dD, -(2.0 + dlnE_int(lna))*dD + 1.5*Om_a_int(lna)*G_int(lna)*D]

    sol = solve_ivp(rhs, [lna_grid[0], lna_grid[-1]],
                    [a_init, a_init],
                    t_eval=lna_grid, method='Radau',
                    rtol=1e-9, atol=1e-11)
    return a_grid, sol.y[0]


def sigma_8_Geff(T_func, G_func, Om, sigma8_LCDM_ref=0.811, Om_LCDM_ref=0.3153):
    a_m, D_m = solve_growth_Geff(T_func, G_func, Om)
    a_p, D_p = solve_growth_Geff(None, None, Om_LCDM_ref)
    return sigma8_LCDM_ref * D_m[-1] / D_p[-1]


def run_sigma8_direct():
    print("\n" + "=" * 76)
    print("Sec. 11 sigma_8 VIA DIRECT GROWTH ODE  (Table 3, left column)")
    print("=" * 76)
    print(f"  Anchor: Planck-LCDM sigma_8 reference = 0.811 at Om = 0.3153")
    print()

    rows = []
    configs = [
        ("LCDM at Om = Om_lock = 0.2907",       None,                                       OM_RECURSIVE),
        ("v1.1 single rung",                    lambda z: T_ladder(z, n_max=1),             OM_RECURSIVE),
        ("Same-sign damped ladder, n_max=2",    lambda z: T_ladder(z, 2, 1/PHI, 'same'),    OM_RECURSIVE),
        ("Same-sign damped ladder, n_max=3",    lambda z: T_ladder(z, 3, 1/PHI, 'same'),    OM_RECURSIVE),
        ("Alt-sign amplified ladder, n_max=2",  lambda z: T_ladder(z, 2, PHI,   'alt'),     OM_RECURSIVE),
    ]
    print(f"  {'model':<45s}  {'sigma_8':>9s}  {'Planck dev':>11s}")
    for label, T, Om in configs:
        s8 = sigma_8(T, Om)
        dev = (s8 - 0.811)/0.006
        print(f"  {label:<45s}  {s8:>9.5f}  {dev:>+11.3f}σ")
        rows.append((label, s8))
    return rows


# ============================================================================
# SECTION 12.  CAMB CROSS-CHECK  (Table 3, right column; OPTIONAL)
# ============================================================================

def run_sigma8_camb():
    print("\n" + "=" * 76)
    print("Sec. 12 sigma_8 VIA CAMB (DarkEnergyPPF)  (Table 3, right column)")
    print("=" * 76)
    try:
        import camb
    except ImportError:
        print("  [SKIPPED: CAMB not available. Install with: pip install camb]")
        return None
    print(f"  CAMB version: {camb.__version__}")
    print(f"  This will run ~5 CAMB calls; takes ~1 minute.")
    print()

    OMBH2_C = 0.02273
    TAU_C   = 0.0544
    AS_C    = 2.1e-9
    NS_C    = 0.9649
    MNU_C   = 0.06

    def compute_w_de_table(T_func, Om, Or):
        a_grid = np.concatenate([
            np.linspace(1.0, 0.5, 600),
            np.linspace(0.5, 0.1, 400)[1:],
            np.geomspace(0.1, 1e-7, 600)[1:],
        ])
        a_grid = np.sort(a_grid)
        z_grid = 1.0/a_grid - 1.0
        OL = 1.0 - Om - Or
        E2_lcdm = Om*(1+z_grid)**3 + Or*(1+z_grid)**4 + OL
        T_vals = T_func(z_grid) if T_func is not None else 1.0
        E2_lvc = E2_lcdm * T_vals**2
        rho_de_eff = E2_lvc - Om*(1+z_grid)**3 - Or*(1+z_grid)**4
        eps = 1e-30
        ln_rho = np.log(np.maximum(rho_de_eff, eps))
        ln_opz = np.log(1.0 + z_grid)
        sort_idx = np.argsort(z_grid)
        ln_rho_s = ln_rho[sort_idx]; ln_opz_s = ln_opz[sort_idx]
        dlnrho = np.gradient(ln_rho_s, ln_opz_s)
        w_eff_s = -1.0 + dlnrho/3.0
        a_sb = 1.0/(1.0 + z_grid[sort_idx])
        idx = np.argsort(a_sb)
        return a_sb[idx], w_eff_s[idx]

    def run(H0, Om, T):
        h = H0/100.0
        omch2 = Om*h*h - OMBH2_C - MNU_C/93.14
        if omch2 <= 0: return None
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=OMBH2_C, omch2=omch2,
                           mnu=MNU_C, omk=0, tau=TAU_C)
        pars.InitPower.set_params(As=AS_C, ns=NS_C)
        pars.set_for_lmax(2500, lens_potential_accuracy=0)
        if T is not None:
            pb = camb.CAMBparams()
            pb.set_cosmology(H0=H0, ombh2=OMBH2_C, omch2=omch2,
                             mnu=MNU_C, omk=0, tau=TAU_C)
            res_b = camb.get_background(pb)
            omega_r = res_b.get_Omega('photon', 0) + res_b.get_Omega('neutrino', 0)
            Om_actual = (OMBH2_C + omch2 + MNU_C/93.14)/h**2
            a_tab, w_tab = compute_w_de_table(T, Om_actual, omega_r)
            order = np.argsort(a_tab); a_tab = a_tab[order]; w_tab = w_tab[order]
            _, uniq = np.unique(a_tab, return_index=True)
            a_tab = a_tab[uniq]; w_tab = w_tab[uniq]
            de = camb.dark_energy.DarkEnergyPPF()
            de.set_w_a_table(a_tab, w_tab)
            pars.DarkEnergy = de
        pars.set_matter_power(redshifts=[0.0], kmax=2.0)
        res = camb.get_results(pars)
        return res.get_sigma8_0()

    print(f"  {'model':<45s}  {'CAMB sigma_8':>13s}")
    configs = [
        ("LCDM at Om = Om_lock",       69.04, None),
        ("v1.1 single rung",           69.06, lambda z: T_ladder(z, 1)),
        ("Same-sign damped, n_max=2",  69.03, lambda z: T_ladder(z, 2, 1/PHI, 'same')),
        ("Same-sign damped, n_max=3",  69.02, lambda z: T_ladder(z, 3, 1/PHI, 'same')),
        ("Alt-sign amplified, n_max=2", 69.15, lambda z: T_ladder(z, 2, PHI, 'alt')),
    ]
    rows = []
    for label, H0, T in configs:
        t0 = time.time()
        s8 = run(H0, OM_RECURSIVE, T)
        print(f"  {label:<45s}  {s8:>13.5f}   [{time.time()-t0:.1f}s]")
        rows.append((label, s8))
    return rows


# ============================================================================
# SECTION 13.  NON-MINIMAL COUPLING  (Table 4)
# ============================================================================

def vartheta_kink_multi(z, n_max=1):
    """Sum of half-twist sine-Gordon kink phases at theta_{c,n} = n ln phi."""
    z = np.asarray(z, dtype=float); theta = np.log(1+z)
    vth = np.zeros_like(theta)
    for n in range(1, n_max+1):
        u = np.clip((theta - n*THETA_C_PHI)/LNPHI_3, -50, 50)
        vth += 8.0 * np.arctan(np.exp(u))
    return vth


def G_eff_natural(z, alpha, n_max=1):
    """G_eff = 1/f, f = exp(-alpha (1 - cos(vartheta/2)))."""
    vth = vartheta_kink_multi(z, n_max)
    return np.exp(alpha * (1.0 - np.cos(vth/2.0)))


def run_alpha_scan():
    print("\n" + "=" * 76)
    print("Sec. 13 NON-MINIMAL COUPLING alpha  (Table 4)")
    print("=" * 76)
    print("  f(vartheta) = exp(-alpha (1 - cos(vartheta/2)))")
    print("  G_eff(z)   = 1/f(vartheta(z))")
    print("  Target: sigma_8(z=0) = 0.811")
    print()

    target = 0.811
    rows = []
    print(f"  {'background':<40s}  {'alpha*':>10s}  {'sigma_8':>9s}")
    for label, T, n_max_kink in [
        ("v1.1 single rung",                lambda z: T_ladder(z, 1),                 1),
        ("Same-sign damped, n_max=2",       lambda z: T_ladder(z, 2, 1/PHI, 'same'),  2),
        ("Same-sign damped, n_max=3",       lambda z: T_ladder(z, 3, 1/PHI, 'same'),  3),
    ]:
        f = lambda a: sigma_8_Geff(T, lambda z: G_eff_natural(z, a, n_max_kink),
                                    OM_RECURSIVE) - target
        a_star = brentq(f, -2.0, 2.0, xtol=1e-8)
        s8 = sigma_8_Geff(T, lambda z: G_eff_natural(z, a_star, n_max_kink),
                          OM_RECURSIVE)
        print(f"  {label:<40s}  {a_star:>+10.5f}  {s8:>9.5f}")
        rows.append((label, a_star, s8))
    print()
    print("  Note: alpha is treated as a phenomenological EFT parameter.")
    print("  No natural-constant identification is assigned.")
    return rows


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LVC final-v1.3 reproduction script")
    parser.add_argument('--skip-pp-included', action='store_true',
        help='Skip PP-included (N=1738) sections (saves ~5 min)')
    parser.add_argument('--skip-camb', action='store_true',
        help='Skip CAMB cross-check (Section 12)')
    parser.add_argument('--save-json', type=str, default=None,
        help='Save results to JSON file')
    args = parser.parse_args()

    print("=" * 76)
    print("LVC final-v1.3 reproduction (LUMENPIXEL, May 2026)")
    print("=" * 76)
    print(f"  phi          = {PHI:.10f}")
    print(f"  ln phi       = {THETA_C_PHI:.10f}")
    print(f"  (ln phi)^3   = {LNPHI_3:.10f}")
    print(f"  (ln phi)^4   = {LNPHI_4:.10f}")
    print(f"  Om recursive = {OM_RECURSIVE:.10f}")

    pp = None
    if not args.skip_pp_included:
        print("\nLoading Pantheon+ data (~33 MB)...")
        pp = load_pantheonplus()
        print(f"  Loaded: {pp['N']} supernovae")

    all_results = {}
    all_results['baselines']    = run_baselines(pp)
    all_results['family_fits']  = run_family_fits(pp)
    all_results['A2_profile']   = run_A2_profile(pp)
    all_results['sigma8_ode']   = run_sigma8_direct()
    if not args.skip_camb:
        all_results['sigma8_camb'] = run_sigma8_camb()
    all_results['alpha_scan']   = run_alpha_scan()

    print("\n" + "=" * 76)
    print("REPRODUCTION COMPLETE.")
    print("=" * 76)

    if args.save_json:
        # JSON-friendly conversion
        def conv(o):
            if isinstance(o, (np.ndarray,)): return o.tolist()
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.integer,)): return int(o)
            return o
        import json as _json
        with open(args.save_json, 'w') as f:
            _json.dump(all_results, f, default=conv, indent=2)
        print(f"Results saved to {args.save_json}")


if __name__ == "__main__":
    main()
