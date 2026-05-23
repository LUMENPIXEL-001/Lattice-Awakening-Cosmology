# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LUMENPIXEL
#
# Released under the MIT License.

"""
LVC final-v1.4 -- Single-file reproduction script
==================================================

Reproduces the numerical results in the final-v1.4 working paper:

    "Lagrangian Variable Cosmology, final-v1.4:
     A Modified-Friedmann Lift of the v1.1 Static Kink
     and Its All-Redshift Behaviour"
                                LUMENPIXEL, May 2026

This is a single self-contained Python file. The required external
dependencies are numpy, pandas, scipy.

The construction recorded in v1.4 substitutes the v1.1 static-kink
coordinate function vartheta_kink(z) into a non-minimal coupling
f(vartheta) = exp(-alpha (1 - cos(vartheta/2))) and lifts the resulting
modulation to the Friedmann equation. The identity
    1 - cos(vartheta_kink/2) = 2 sech^2((theta - theta_c)/w),  theta = ln(1+z)
that follows from the v1.1 kink identity makes f(vartheta_kink(z)) a
closed function of z. The Friedmann equation then yields
    T(z) = H(z)/H_LCDM(z) = exp[ alpha_b sech^2((theta - theta_c)/w) ],
with locks theta_c = ln phi, w = (ln phi)^3, A = (ln phi)^4 imported
from v1.1.

The likelihood pipeline (Sections 1-7) is identical to v1.3, which is
reproduced as part of the sanity-check phase (Section 8).

Reproduces (key paper numbers, PP-excluded N=37 unless noted):
---------------------------------------------------------------
Section 8  Sanity-check baselines (paper Tables 2, 3)
    LCDM PP-excluded                (k=3)            chi^2 = 78.962
    v1.1 sech^2 PP-excluded         (k=2)            chi^2 = 56.134
    v1.4 Eq.(6), alpha_b=A          (k=2)            chi^2 = 56.090
    v1.4 Eq.(6), alpha_b free       (k=3)            chi^2 = 56.086
        (best-fit alpha_b = 0.0541, A = 0.0536)

    PP-included (N=1738):
    LCDM PP-included                (k=3)            chi^2 = 1624.848
    v1.1 sech^2 PP-included         (k=2)            chi^2 = 1601.999
    v1.4 Eq.(6), alpha_b=A          (k=2)            chi^2 = 1601.964
    v1.4 Eq.(6), alpha_b free       (k=3)            chi^2 = 1601.958
        (best-fit alpha_b = 0.0542, A = 0.0536)

Section 9  Distance-prior contribution                      (paper Table 4)
Section 10 All-redshift behaviour                           (paper Table 1)
Section 11 f sigma_8 fit                                    (paper Table 5)
    LCDM at Om = Om_lock                            chi^2 = 12.181
    v1.4, alpha_b=A, alpha_g=0                      chi^2 = 13.776
    v1.4, alpha_b=A, alpha_g free  -> 0.107         chi^2 = 12.243
    v1.4, alpha_b + alpha_g free                    chi^2 = 11.998
Section 12 S_8 table                                        (paper Table 6)

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
    python lvc_v14_reproduce.py                  # default (full run)
    python lvc_v14_reproduce.py --skip-pp-included   # PP-excluded only
    python lvc_v14_reproduce.py --skip-fsigma8   # skip Section 11
    python lvc_v14_reproduce.py --help           # full options

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


def T_v11(z, A=LNPHI_4):
    """v1.1 single rung: T(z) = 1 + A sech^2((theta - theta_c)/w),
    theta = ln(1+z), theta_c = ln phi, w = (ln phi)^3.
    """
    z = np.asarray(z, dtype=float)
    theta = np.log(1.0 + z)
    arg = np.clip((theta - THETA_C_PHI) / LNPHI_3, -50.0, 50.0)
    return 1.0 + A / np.cosh(arg) ** 2


def T_v14(z, alpha_b=LNPHI_4):
    """v1.4 Eq.(6):  T(z) = exp[ alpha_b sech^2((theta - theta_c)/w) ].

    Default alpha_b = A = (ln phi)^4 (v1.1 small-amplitude match).
    """
    z = np.asarray(z, dtype=float)
    theta = np.log(1.0 + z)
    arg = np.clip((theta - THETA_C_PHI) / LNPHI_3, -50.0, 50.0)
    return np.exp(alpha_b / np.cosh(arg) ** 2)


def sech2_envelope(z):
    """sech^2((theta - theta_c)/w) at theta = ln(1+z)."""
    z = np.asarray(z, dtype=float)
    theta = np.log(1.0 + z)
    arg = np.clip((theta - THETA_C_PHI) / LNPHI_3, -50.0, 50.0)
    return 1.0 / np.cosh(arg) ** 2


def vartheta_kink(z, n_max=1):
    """v1.1 static kink coordinate function, retained from v1.3.
    For n_max > 1 this is the v1.3 multi-rung extension; v1.4 uses n_max=1.
    """
    z = np.asarray(z, dtype=float)
    theta = np.log(1.0 + z)
    if n_max == 1:
        return 8.0 * np.arctan(np.exp((1.0/LNPHI_3) * (theta - THETA_C_PHI)))
    # multi-rung (v1.3 extension; preserved for cross-checks)
    vth = np.zeros_like(theta)
    for n in range(1, n_max + 1):
        arg = (1.0/LNPHI_3) * (theta - n*THETA_C_PHI)
        vth = vth + 8.0 * np.arctan(np.exp(arg))
    return vth


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
# SECTION 5.  BAO DATA TABLES (identical to v1.3)
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
# SECTION 6.  DISTANCE PRIORS (identical to v1.3)
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
# SECTION 7.  COMBINED LIKELIHOODS (identical to v1.3)
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
# SECTION 8.  BASELINES AND V1.4 BACKGROUND FITS  (paper Tables 2, 3)
# ============================================================================

def _de_nm(cost, bounds, seed=42, maxiter=80, popsize=12, nm_iter=3000):
    de = differential_evolution(cost, bounds, seed=seed, tol=1e-7,
                                maxiter=maxiter, polish=False, popsize=popsize)
    nm = minimize(cost, de.x, method='Nelder-Mead',
                  options=dict(xatol=1e-8, fatol=1e-8, maxiter=nm_iter,
                               adaptive=True))
    return float(nm.fun), nm.x.tolist()


def run_baselines(pp=None):
    print("=" * 76)
    print("Sec. 8 BACKGROUND FITS  (paper Tables 2, 3)")
    print("=" * 76)

    results = {}

    # ----- PP-excluded -----
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
    print("  (paper value: chi^2 = 78.962)")
    results['lcdm_ppex'] = dict(chi2=chi_lcdm, H0=x_lcdm[0], Om=x_lcdm[1], Ob=x_lcdm[2])

    def cost_v11(p):
        H0, Ob = p
        if not (60<=H0<=80 and 0.018<=Ob<=0.028): return 1e10
        c = chi2_no_pp(H0, OM_RECURSIVE, Ob, T_v11)
        return c if np.isfinite(c) else 1e10

    print("\n[B] v1.1 sech^2 locked PP-excluded (k=2) ...")
    t0 = time.time()
    chi_v11, x_v11 = _de_nm(cost_v11, [(60,80),(0.020,0.026)],
                            maxiter=50, popsize=10)
    bic_lcdm = chi_lcdm + 3*np.log(N_PP_EXCLUDED)
    bic_v11 = chi_v11 + 2*np.log(N_PP_EXCLUDED)
    print(f"  chi^2 = {chi_v11:.4f}  H0={x_v11[0]:.3f}  Ob={x_v11[1]:.5f}")
    print(f"  dBIC vs LCDM = {bic_v11-bic_lcdm:+.3f}  [{time.time()-t0:.1f}s]")
    print("  (paper value: chi^2 = 56.134, dBIC = -26.44)")
    results['v11_ppex'] = dict(chi2=chi_v11, H0=x_v11[0], Ob=x_v11[1],
                                dBIC=bic_v11-bic_lcdm)

    def cost_v14_lock(p):
        H0, Ob = p
        if not (60<=H0<=80 and 0.018<=Ob<=0.028): return 1e10
        T = lambda z: T_v14(z, alpha_b=LNPHI_4)
        c = chi2_no_pp(H0, OM_RECURSIVE, Ob, T)
        return c if np.isfinite(c) else 1e10

    print("\n[C] v1.4 Eq.(6) alpha_b = A locked PP-excluded (k=2) ...")
    t0 = time.time()
    chi_v14, x_v14 = _de_nm(cost_v14_lock, [(60,80),(0.020,0.026)],
                            maxiter=50, popsize=10)
    bic_v14 = chi_v14 + 2*np.log(N_PP_EXCLUDED)
    print(f"  chi^2 = {chi_v14:.4f}  H0={x_v14[0]:.3f}  Ob={x_v14[1]:.5f}")
    print(f"  dBIC vs LCDM = {bic_v14-bic_lcdm:+.3f}  [{time.time()-t0:.1f}s]")
    print("  (paper value: chi^2 = 56.090, dBIC = -26.48)")
    results['v14_lock_ppex'] = dict(chi2=chi_v14, H0=x_v14[0], Ob=x_v14[1],
                                     dBIC=bic_v14-bic_lcdm)

    def cost_v14_free(p):
        H0, Ob, ab = p
        if not (60<=H0<=80 and 0.018<=Ob<=0.028 and -0.05<=ab<=0.30):
            return 1e10
        T = lambda z: T_v14(z, alpha_b=ab)
        c = chi2_no_pp(H0, OM_RECURSIVE, Ob, T)
        return c if np.isfinite(c) else 1e10

    print("\n[D] v1.4 Eq.(6) alpha_b free PP-excluded (k=3) ...")
    t0 = time.time()
    chi_v14f, x_v14f = _de_nm(cost_v14_free,
        [(60,80),(0.020,0.026),(-0.05,0.30)], maxiter=80, popsize=15)
    bic_v14f = chi_v14f + 3*np.log(N_PP_EXCLUDED)
    print(f"  chi^2 = {chi_v14f:.4f}  H0={x_v14f[0]:.3f}  "
          f"Ob={x_v14f[1]:.5f}  alpha_b={x_v14f[2]:.5f}")
    print(f"  dBIC vs LCDM = {bic_v14f-bic_lcdm:+.3f}  [{time.time()-t0:.1f}s]")
    print(f"  (paper value: chi^2 = 56.086, alpha_b = 0.0541, A = {LNPHI_4:.4f})")
    results['v14_free_ppex'] = dict(chi2=chi_v14f, H0=x_v14f[0], Ob=x_v14f[1],
                                     alpha_b=x_v14f[2], dBIC=bic_v14f-bic_lcdm)

    if pp is None:
        return results

    # ----- PP-included -----
    def cost_lcdm_pp(p):
        H0, Om, Ob = p
        if not (60<=H0<=80 and 0.18<=Om<=0.45 and 0.018<=Ob<=0.028): return 1e10
        c = chi2_full(H0, Om, Ob, None, pp)
        return c if np.isfinite(c) else 1e10

    print("\n[E] LCDM PP-included (k=3) ...")
    t0 = time.time()
    chi_lcdm_pp, x_lcdm_pp = _de_nm(cost_lcdm_pp,
        [(65,75),(0.25,0.35),(0.020,0.024)], maxiter=60, popsize=10)
    print(f"  chi^2 = {chi_lcdm_pp:.4f}  H0={x_lcdm_pp[0]:.3f}  "
          f"Om={x_lcdm_pp[1]:.4f}  [{time.time()-t0:.1f}s]")
    print("  (paper value: chi^2 = 1624.848)")
    results['lcdm_ppin'] = dict(chi2=chi_lcdm_pp, H0=x_lcdm_pp[0],
                                 Om=x_lcdm_pp[1], Ob=x_lcdm_pp[2])

    def cost_v11_pp(p):
        H0, Ob = p
        if not (65<=H0<=75 and 0.018<=Ob<=0.028): return 1e10
        c = chi2_full(H0, OM_RECURSIVE, Ob, T_v11, pp)
        return c if np.isfinite(c) else 1e10

    print("\n[F] v1.1 sech^2 locked PP-included (k=2) ...")
    t0 = time.time()
    chi_v11_pp, x_v11_pp = _de_nm(cost_v11_pp,
        [(65,75),(0.020,0.024)], maxiter=40, popsize=10)
    bic_lcdm_pp = chi_lcdm_pp + 3*np.log(N_PP_INCLUDED)
    bic_v11_pp = chi_v11_pp + 2*np.log(N_PP_INCLUDED)
    print(f"  chi^2 = {chi_v11_pp:.4f}  H0={x_v11_pp[0]:.3f}  Ob={x_v11_pp[1]:.5f}")
    print(f"  dBIC vs LCDM = {bic_v11_pp-bic_lcdm_pp:+.3f}  [{time.time()-t0:.1f}s]")
    print("  (paper value: chi^2 = 1601.999, dBIC = -30.31)")
    results['v11_ppin'] = dict(chi2=chi_v11_pp, H0=x_v11_pp[0],
                                Ob=x_v11_pp[1], dBIC=bic_v11_pp-bic_lcdm_pp)

    def cost_v14_lock_pp(p):
        H0, Ob = p
        if not (65<=H0<=75 and 0.018<=Ob<=0.028): return 1e10
        T = lambda z: T_v14(z, alpha_b=LNPHI_4)
        c = chi2_full(H0, OM_RECURSIVE, Ob, T, pp)
        return c if np.isfinite(c) else 1e10

    print("\n[G] v1.4 Eq.(6) alpha_b = A locked PP-included (k=2) ...")
    t0 = time.time()
    chi_v14_pp, x_v14_pp = _de_nm(cost_v14_lock_pp,
        [(65,75),(0.020,0.024)], maxiter=40, popsize=10)
    bic_v14_pp = chi_v14_pp + 2*np.log(N_PP_INCLUDED)
    print(f"  chi^2 = {chi_v14_pp:.4f}  H0={x_v14_pp[0]:.3f}  Ob={x_v14_pp[1]:.5f}")
    print(f"  dBIC vs LCDM = {bic_v14_pp-bic_lcdm_pp:+.3f}  [{time.time()-t0:.1f}s]")
    print("  (paper value: chi^2 = 1601.964, dBIC = -30.34)")
    results['v14_lock_ppin'] = dict(chi2=chi_v14_pp, H0=x_v14_pp[0],
                                     Ob=x_v14_pp[1], dBIC=bic_v14_pp-bic_lcdm_pp)

    def cost_v14_free_pp(p):
        H0, Ob, ab = p
        if not (65<=H0<=75 and 0.018<=Ob<=0.028 and -0.05<=ab<=0.30):
            return 1e10
        T = lambda z: T_v14(z, alpha_b=ab)
        c = chi2_full(H0, OM_RECURSIVE, Ob, T, pp)
        return c if np.isfinite(c) else 1e10

    print("\n[H] v1.4 Eq.(6) alpha_b free PP-included (k=3) ...")
    t0 = time.time()
    chi_v14f_pp, x_v14f_pp = _de_nm(cost_v14_free_pp,
        [(65,75),(0.020,0.024),(-0.05,0.30)], maxiter=80, popsize=15)
    bic_v14f_pp = chi_v14f_pp + 3*np.log(N_PP_INCLUDED)
    print(f"  chi^2 = {chi_v14f_pp:.4f}  H0={x_v14f_pp[0]:.3f}  "
          f"Ob={x_v14f_pp[1]:.5f}  alpha_b={x_v14f_pp[2]:.5f}")
    print(f"  dBIC vs LCDM = {bic_v14f_pp-bic_lcdm_pp:+.3f}  [{time.time()-t0:.1f}s]")
    print(f"  (paper value: chi^2 = 1601.958, alpha_b = 0.0542)")
    results['v14_free_ppin'] = dict(chi2=chi_v14f_pp, H0=x_v14f_pp[0],
                                     Ob=x_v14f_pp[1], alpha_b=x_v14f_pp[2],
                                     dBIC=bic_v14f_pp-bic_lcdm_pp)

    return results


# ============================================================================
# SECTION 9.  DISTANCE-PRIOR CONTRIBUTION  (paper Table 4)
# ============================================================================

def run_distance_prior(baseline_results):
    print("\n" + "=" * 76)
    print("Sec. 9 DISTANCE-PRIOR CONTRIBUTION  (paper Table 4)")
    print("=" * 76)
    print(f"\n  Planck DP central: R={R_PLANCK:.4f}+/-{SIG_R:.4f}, "
          f"lA={LA_PLANCK:.3f}+/-{SIG_LA:.3f},")
    print(f"                     Ob h^2 = {WB_PLANCK:.5f}+/-{SIG_WB:.5f}")

    # use best-fit parameters from PP-included baselines if available
    pp_data = baseline_results.get('lcdm_ppin')
    if pp_data is None:
        print("\n  (PP-included baselines not available; using nominal parameters)")
        configs = [
            ("LCDM             ", 70.40, 0.2764, 0.02286, None),
            ("v1.1 sech^2 lock ", 69.06, OM_RECURSIVE, 0.02275, T_v11),
            ("v1.4 Eq.(6)      ", 69.06, OM_RECURSIVE, 0.02276,
                lambda z: T_v14(z, alpha_b=LNPHI_4)),
        ]
    else:
        configs = [
            ("LCDM             ",
                baseline_results['lcdm_ppin']['H0'],
                baseline_results['lcdm_ppin']['Om'],
                baseline_results['lcdm_ppin']['Ob'], None),
            ("v1.1 sech^2 lock ",
                baseline_results['v11_ppin']['H0'],
                OM_RECURSIVE,
                baseline_results['v11_ppin']['Ob'], T_v11),
            ("v1.4 Eq.(6) lock ",
                baseline_results['v14_lock_ppin']['H0'],
                OM_RECURSIVE,
                baseline_results['v14_lock_ppin']['Ob'],
                lambda z: T_v14(z, alpha_b=LNPHI_4)),
        ]

    print(f"\n  {'Model':<20} {'H0':>7} {'chi2_DP':>10} {'R':>10} {'lA':>10}")
    rows = []
    for name, H0, Om, Ob, Tf in configs:
        chi_dp, R_val, lA_val = chi2_DP(H0, Om, Ob, Tf)
        print(f"  {name:<20} {H0:>7.2f} {chi_dp:>10.3f} {R_val:>10.4f} {lA_val:>10.3f}")
        rows.append(dict(model=name.strip(), H0=H0, chi2_DP=chi_dp,
                         R=R_val, lA=lA_val))
    print("\n  (paper Table 4: LCDM 21.86, v1.1 12.15, v1.4 12.28)")
    return rows


# ============================================================================
# SECTION 10.  ALL-REDSHIFT BEHAVIOUR  (paper Table 1)
# ============================================================================

def run_alltime():
    print("\n" + "=" * 76)
    print("Sec. 10 ALL-REDSHIFT BEHAVIOUR  (paper Table 1)")
    print("=" * 76)
    print(f"\n  Eq.(6): T(z) = exp[ alpha_b sech^2((theta-theta_c)/w) ]")
    print(f"  alpha_b = A = (ln phi)^4 = {LNPHI_4:.6f}")
    print()
    print(f"  {'z':>10}  {'epoch':<20}  {'T(z) - 1':>16}")
    print("  " + "-"*52)

    rows = []
    epochs = [
        (0.0,    "today"),
        (1/PHI,  "peak (z = 1/phi)"),
        (1.0,    "mid-z BAO"),
        (2.33,   "Ly-alpha BAO"),
        (10.0,   "high-z"),
        (30.0,   "(envelope decay)"),
        (1090.0, "recombination"),
        (1e9,    "nucleosynthesis"),
    ]
    for z, epoch in epochs:
        T = float(T_v14(np.array([z]))[0])
        Tm1 = T - 1.0
        if abs(Tm1) < 1e-300:
            rep = "underflow"
        else:
            rep = f"{Tm1:>16.3e}"
        print(f"  {z:>10.3g}  {epoch:<20}  {rep:>16}")
        rows.append(dict(z=z, epoch=epoch, T_minus_1=Tm1))
    print()
    print("  Reduction to LCDM at numerical precision for z >~ 30.")
    return rows


# ============================================================================
# SECTION 11.  GROWTH ODE AND f sigma_8 FIT  (paper Table 5)
# ============================================================================

OR_FRAC_DEFAULT = OMEGA_R_H2 / 0.70**2


def _growth_rhs(N, y, alpha_b, alpha_g, Om, Or_frac):
    """
    D'' + (2 + d ln H_LVC/dN) D' = (3/2) Omega_m(a)_LVC (G_eff/G) D

    H_LVC = H_LCDM * T(z),  T(z) given by Eq.(6).
    """
    D, Dp = y
    a = np.exp(N)
    z = 1.0/a - 1.0
    th = np.log(1.0 + z)
    arg = np.clip((th - THETA_C_PHI)/LNPHI_3, -50.0, 50.0)
    s2 = 1.0/np.cosh(arg)**2
    tanh_u = np.tanh(arg)

    OL = 1.0 - Om - Or_frac
    H2_lcdm = Om*a**(-3) + Or_frac*a**(-4) + OL
    dH2_lcdm_dN = -3.0*Om*a**(-3) - 4.0*Or_frac*a**(-4)
    dlnH_lcdm = 0.5*dH2_lcdm_dN/H2_lcdm

    # T(z) = exp[ alpha_b s2 ]; theta = -N,
    # d sech^2(u)/dN with u = (theta-theta_c)/w, dtheta/dN = -1
    #   d/dN [sech^2(u)] = -2 sech^2 tanh(u) * (-1/w) = (2/w) s2 tanh(u)
    dlnH_lvc = dlnH_lcdm + alpha_b*(2.0/LNPHI_3)*s2*tanh_u

    T2 = np.exp(2.0*alpha_b*s2)
    Omega_m_a_lvc = Om*a**(-3) / (H2_lcdm * T2)

    Geff = np.exp(2.0*alpha_g*s2)

    Dpp = -(2.0 + dlnH_lvc)*Dp + 1.5*Omega_m_a_lvc*Geff*D
    return [Dp, Dpp]


def solve_growth(alpha_b, alpha_g, Om=OM_RECURSIVE, Or_frac=OR_FRAC_DEFAULT,
                  N_init=-6.0, N_end=0.0, n_points=4000):
    a_init = np.exp(N_init)
    N_eval = np.linspace(N_init, N_end, n_points)
    sol = solve_ivp(_growth_rhs, [N_init, N_end], [a_init, a_init],
                    args=(alpha_b, alpha_g, Om, Or_frac),
                    method='DOP853', t_eval=N_eval,
                    rtol=1e-11, atol=1e-13)
    return sol


# fsigma_8 compilation: (z, fsig8_obs, sigma, source label)
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
SIGMA8_PLANCK = 0.811


def fsigma8_predict(z_array, alpha_b, alpha_g, Om=OM_RECURSIVE,
                     Or_frac=OR_FRAC_DEFAULT):
    """
    f sigma_8(z) = f(z) * D(z)/D(0)_LVC * sigma_8(z=0, LVC)
    with sigma_8(z=0, LVC) = sigma_8_Planck * D_LVC(0) / D_LCDM_ref(0),
    where the reference is LCDM at the *same* Omega_m as the LVC model
    (i.e. Om = v1.1 lock by default).

    The choice of reference fixes how the primordial amplitude is propagated;
    we use the reference that matches paper Table 5 (Om = v1.1 lock).
    """
    sol = solve_growth(alpha_b, alpha_g, Om, Or_frac)
    N_arr = sol.t
    D_arr = sol.y[0]
    Dp_arr = sol.y[1]
    f_arr = Dp_arr / D_arr
    D0_lvc = D_arr[-1]

    sol_norm = solve_growth(0.0, 0.0, Om=Om, Or_frac=Or_frac)
    D0_norm = sol_norm.y[0, -1]
    sigma8_0 = SIGMA8_PLANCK * D0_lvc / D0_norm

    z_arr = np.asarray(z_array, dtype=float)
    N_targets = -np.log(1.0 + z_arr)
    D_z = np.interp(N_targets, N_arr, D_arr)
    f_z = np.interp(N_targets, N_arr, f_arr)
    return f_z * (D_z / D0_lvc) * sigma8_0, sigma8_0


def chi2_fsigma8(alpha_b, alpha_g, Om=OM_RECURSIVE):
    z_vals = np.array([d[0] for d in FSIG8_DATA])
    fs_obs = np.array([d[1] for d in FSIG8_DATA])
    fs_err = np.array([d[2] for d in FSIG8_DATA])
    fs_pred, _ = fsigma8_predict(z_vals, alpha_b, alpha_g, Om)
    return float(np.sum((fs_obs - fs_pred)**2 / fs_err**2))


def run_fsigma8():
    print("\n" + "=" * 76)
    print("Sec. 11 f sigma_8 FIT  (paper Table 5)")
    print("=" * 76)
    print(f"\n  Data set: N = {len(FSIG8_DATA)} f sigma_8 measurements,")
    print(f"  z in [{FSIG8_DATA[0][0]:.3f}, {max(d[0] for d in FSIG8_DATA):.3f}]")
    print(f"  Normalisation: sigma_8(LCDM, Om=0.315) = {SIGMA8_PLANCK}")

    results = []

    # LCDM at Om = v1.1 lock
    c = chi2_fsigma8(0.0, 0.0)
    _, s8 = fsigma8_predict([0.0], 0.0, 0.0)
    print(f"\n  [A] LCDM at Om = v1.1 lock           chi^2 = {c:.3f}  "
          f"sigma_8(0) = {s8:.4f}")
    print(f"      (paper value: chi^2 = 12.181)")
    results.append(dict(model='LCDM_om_lock', k=0, chi2=c, sigma8=s8))

    # v1.4 alpha_b = A, alpha_g = 0
    c = chi2_fsigma8(LNPHI_4, 0.0)
    _, s8 = fsigma8_predict([0.0], LNPHI_4, 0.0)
    print(f"  [B] v1.4 alpha_b=A, alpha_g=0        chi^2 = {c:.3f}  "
          f"sigma_8(0) = {s8:.4f}")
    print(f"      (paper value: chi^2 = 13.776)")
    results.append(dict(model='v14_no_growth', k=0, chi2=c, sigma8=s8))

    # v1.4 alpha_b = A, alpha_g free
    def cost1(p):
        return chi2_fsigma8(LNPHI_4, p[0])
    print(f"\n  [C] v1.4 alpha_b=A, alpha_g free (k=1) ...")
    t0 = time.time()
    res = differential_evolution(cost1, [(-0.2, 0.5)], seed=42, tol=1e-7,
                                  maxiter=100, polish=True, popsize=10)
    ag = res.x[0]
    _, s8 = fsigma8_predict([0.0], LNPHI_4, ag)
    print(f"      alpha_g = {ag:.5f}    chi^2 = {res.fun:.3f}  "
          f"sigma_8(0) = {s8:.4f}  [{time.time()-t0:.1f}s]")
    print(f"      (paper value: chi^2 = 12.243, alpha_g = 0.107)")
    results.append(dict(model='v14_alpha_g_free', k=1, chi2=float(res.fun),
                        alpha_g=ag, sigma8=s8))

    # v1.4 alpha_b + alpha_g free
    def cost2(p):
        ab, ag = p
        return chi2_fsigma8(ab, ag)
    print(f"\n  [D] v1.4 alpha_b + alpha_g free (k=2) ...")
    t0 = time.time()
    res = differential_evolution(cost2, [(-0.05, 0.30), (-0.20, 0.50)],
                                  seed=42, tol=1e-7, maxiter=150,
                                  polish=True, popsize=15)
    ab, ag = res.x
    _, s8 = fsigma8_predict([0.0], ab, ag)
    print(f"      alpha_b = {ab:.5f}  alpha_g = {ag:.5f}  "
          f"chi^2 = {res.fun:.3f}  sigma_8(0) = {s8:.4f}  [{time.time()-t0:.1f}s]")
    print(f"      (paper value: chi^2 = 11.998)")
    results.append(dict(model='v14_both_free', k=2, chi2=float(res.fun),
                        alpha_b=ab, alpha_g=ag, sigma8=s8))

    # residuals at the alpha_g = 0.107 fit
    z_vals = np.array([d[0] for d in FSIG8_DATA])
    fs_obs = np.array([d[1] for d in FSIG8_DATA])
    fs_err = np.array([d[2] for d in FSIG8_DATA])
    fs_pred, _ = fsigma8_predict(z_vals, LNPHI_4, results[2]['alpha_g'])
    pulls = (fs_obs - fs_pred) / fs_err
    print(f"\n  Residual pulls at alpha_b = A, alpha_g = {results[2]['alpha_g']:.4f}:")
    print(f"    mean pull = {np.mean(pulls):+.3f}")
    print(f"    rms pull  = {np.sqrt(np.mean(pulls**2)):.3f}")
    print(f"    max |pull| = {np.max(np.abs(pulls)):.2f}")

    return results


# ============================================================================
# SECTION 12.  S_8 TABLE  (paper Table 6)
# ============================================================================

def run_S8():
    print("\n" + "=" * 76)
    print("Sec. 12 S_8 TABLE  (paper Table 6)")
    print("=" * 76)
    print("\n  S_8 = sigma_8 * sqrt(Om / 0.3)")
    print("  Normalisation: sigma_8(LCDM, Om=0.315) = 0.811 (Planck reference);")
    print("  growth ratio computed relative to this reference for each row.")

    # baseline: sigma_8(LCDM, Om=0.315) = 0.811 (Planck reference)
    sol_norm = solve_growth(0.0, 0.0, Om=0.315)
    D0_norm = sol_norm.y[0, -1]

    def S8_for(alpha_b, alpha_g, Om=OM_RECURSIVE):
        sol = solve_growth(alpha_b, alpha_g, Om=Om)
        D0 = sol.y[0, -1]
        s8 = SIGMA8_PLANCK * D0 / D0_norm
        return s8, s8 * np.sqrt(Om/0.3)

    print(f"\n  {'Construction':<48} {'sigma_8(0)':>11} {'S_8':>9}")
    print("  " + "-"*70)

    rows = []
    # LCDM at Planck Om
    s8, S8 = S8_for(0.0, 0.0, Om=0.315)
    print(f"  {'LCDM (Om=0.315, Planck normalisation)':<48} {s8:>11.4f} {S8:>9.4f}")
    rows.append(dict(label='LCDM_Planck_Om', sigma8=s8, S8=S8))

    # LCDM at v1.1 lock Om
    s8, S8 = S8_for(0.0, 0.0)
    print(f"  {'LCDM (Om at v1.1 lock = 0.291)':<48} {s8:>11.4f} {S8:>9.4f}")
    rows.append(dict(label='LCDM_lock_Om', sigma8=s8, S8=S8))

    # v1.4 at various alpha_g
    for ag, tag in [(0.0,    'alpha_g = 0'),
                     (0.107,  'alpha_g = 0.107  (fsigma8 best)'),
                     (0.226,  'alpha_g = 0.226  (v1.3 lock)')]:
        s8, S8 = S8_for(LNPHI_4, ag)
        print(f"  {'v1.4 alpha_b=A, ' + tag:<48} {s8:>11.4f} {S8:>9.4f}")
        rows.append(dict(label=f'v14_ag_{ag}', sigma8=s8, S8=S8))

    print("\n  Measurement central values (for comparison):")
    print(f"    WL combined (DES Y3, KiDS-1000, HSC Y3, DES+KiDS):  0.778 +/- 0.010")
    print(f"    CMB primary (Planck 2018):                          0.832 +/- 0.013")
    print(f"    CMB lensing (ACT DR6 + Planck 2018 lensing):        0.818 +/- 0.015")
    return rows


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LVC final-v1.4 reproduction script")
    parser.add_argument('--skip-pp-included', action='store_true',
        help='Skip PP-included (N=1738) sections (saves ~5 min)')
    parser.add_argument('--skip-fsigma8', action='store_true',
        help='Skip Section 11 (f sigma_8 fit)')
    parser.add_argument('--save-json', type=str, default=None,
        help='Save results to JSON file')
    args = parser.parse_args()

    print("=" * 76)
    print("LVC final-v1.4 reproduction (LUMENPIXEL, May 2026)")
    print("=" * 76)
    print(f"  phi             = {PHI:.10f}")
    print(f"  ln phi          = {THETA_C_PHI:.10f}")
    print(f"  (ln phi)^3 = w  = {LNPHI_3:.10f}")
    print(f"  (ln phi)^4 = A  = {LNPHI_4:.10f}")
    print(f"  Om recursive    = {OM_RECURSIVE:.10f}")
    print()
    print("  Background modulation:")
    print("    v1.1 form:  T(z) = 1 + A sech^2((theta-theta_c)/w)")
    print("    v1.4 Eq.6:  T(z) = exp[ alpha_b sech^2((theta-theta_c)/w) ]")

    pp = None
    if not args.skip_pp_included:
        print("\nLoading Pantheon+ data (~33 MB)...")
        pp = load_pantheonplus()
        print(f"  Loaded: {pp['N']} supernovae")

    all_results = {}
    all_results['baselines']      = run_baselines(pp)
    all_results['distance_prior'] = run_distance_prior(all_results['baselines'])
    all_results['alltime']        = run_alltime()
    if not args.skip_fsigma8:
        all_results['fsigma8']    = run_fsigma8()
    all_results['S8']             = run_S8()

    print("\n" + "=" * 76)
    print("REPRODUCTION COMPLETE.")
    print("=" * 76)

    if args.save_json:
        def conv(o):
            if isinstance(o, (np.ndarray,)): return o.tolist()
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.integer,)): return int(o)
            return o
        with open(args.save_json, 'w') as f:
            json.dump(all_results, f, default=conv, indent=2)
        print(f"Results saved to {args.save_json}")


if __name__ == "__main__":
    main()
