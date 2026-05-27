"""
LVC v16 — Single-file reproduction script (self-contained)
===========================================================

Reproduces all numerical results in:
  "Combined DR1+DR2 BAO Fit Identifies a New Lock Candidate (LockPhi)
   at z_c ≈ 1/φ — LVC v16 Working Paper"

This is a single self-contained Python file with no LVC-specific
dependencies. All constants, BAO data, distance-prior calibration,
PP loader, and fitter are inlined from v15 conventions.

External data
-------------
The Pantheon+ data (~33 MB) and covariance must be available locally:
    Pantheon+SH0ES.dat
    Pantheon+SH0ES_STAT+SYS.cov

If absent, the script offers automatic git-based download from
github.com/PantheonPlusSH0ES/DataRelease (sub-directory
Pantheon+_Data/4_DISTANCES_AND_COVAR).

Set PANTHEONPLUS_DIR environment variable, or place files in ./PP_data/

What this script reproduces
---------------------------
1. Combined (DR1 + DR2) BAO likelihood at N = 1738.
2. LockB, LockC, LockPhi-A, LockPhi-B, free-Gaussian fits on the combined
   likelihood (§2-§5.1 of v16 working paper).
3. z_c natural-constant scan at fixed w = e/(5π) (§5.2).
4. 5x5 χ² landscape on (z_c, w) (§5.3).
5. K-A multi-LOO on combined BAO, with labels common to DR1 and DR2
   removed from both releases simultaneously (§6.1).
6. Free-Ω_Λ test on the combined likelihood (§6.2).
7. fσ8 joint analysis with linear growth ODE on locked H(z) (§6.3).
8. BBN-strict r_d = 147.05 ± 0.3 Mpc prior (§6.4).
9. χ² decomposition by data block, BAO point-by-point, A-profile under
   simultaneous LRG1+LRG2 removal (§6.5).
10. DR3 prior predictions for D_M/r_d, D_H/r_d at z=0.510, 0.706 (§7).

Approximate runtime: 90-120 minutes on a single CPU core.

Usage
-----
    python lvc_v16_reproduce.py                # full reproduction
    python lvc_v16_reproduce.py --baseline     # §2-§5.1 only (~10 min)
    python lvc_v16_reproduce.py --ladder       # §5.2 z_c ladder (~25 min)
    python lvc_v16_reproduce.py --landscape    # §5.3 2D landscape (~10 min)
    python lvc_v16_reproduce.py --ka           # §6.1 K-A combined (~25 min)
    python lvc_v16_reproduce.py --kills        # §6.2-§6.4 (~15 min)
    python lvc_v16_reproduce.py --decomp       # §6.5 (~5 min)
    python lvc_v16_reproduce.py --predict      # §7 (<1 min, no fits)

Tabulated chi^2 values reproduce within <= 0.05.
Tabulated DeltaBIC values reproduce within <= 0.10.

Author: LUMENPIXEL (independent researcher)
Computational assistance: Claude (Anthropic)
May 2026
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.integrate import quad, solve_ivp
from scipy.optimize import differential_evolution, minimize, minimize_scalar


# ============================================================================
# SECTION 1: CONSTANTS (from v14 / v15 conventions)
# ============================================================================

C_KMS         = 299_792.458
THETA_STAR    = 0.010409
SIG_THETA_STAR = 3.1e-5
Z_STAR        = 1090.0
H0_SHOES      = 73.04
SIG_H0_SHOES  = 1.04

# Radiation density (Tcmb=2.7255 K, 3.046 neutrino species)
OMEGA_GAMMA_H2 = 2.473e-5
NEFF_STD       = 3.046
OMEGA_R_H2     = OMEGA_GAMMA_H2 * (1 + 7/8 * (4/11)**(4/3) * NEFF_STD)

# Pi and Phi
PI = np.pi
PHI = (1 + np.sqrt(5)) / 2.0
INV_PHI = 1.0 / PHI               # 0.61803...
W_PHI_A = np.e / (5 * PI)         # 0.17305  (LockPhi-A)
W_PHI_B = PI / 18.0               # 0.17453  (LockPhi-B)

# v15 lock parameters
ZC_LOCKB = 1.0 / np.sqrt(PI)      # 0.56419
ZC_LOCKC = 4.0 / 7.0              # 0.57143
W_LOCKBC = 2.0 / (5 * PI)         # 0.12732

# v14 C1 lock (for reference/context)
ZC_C1 = 1.0 / np.sqrt(3.0) + 1.0 / (3 * PI)   # 0.68345
W_C1  = 2.0 / (3 * PI)                         # 0.21221

# Planck 2018 compressed distance priors (Chen, Huang & Wang 2019,
#   Table 1, "TT,TE,EE+lowE+lensing")
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

# Combined-likelihood total measurement count
# 1701 (PP) + 13 (DR2 BAO) + 11 (DR1 BAO) + 2 (BOSS) + 7 (eBOSS) + 3 (DP) + 1 (SH0ES)
N_COMBINED = 1701 + 13 + 11 + 2 + 7 + 3 + 1   # = 1738


# ============================================================================
# SECTION 2: PP DATA LOADER (with auto-download)
# ============================================================================

PP_DIR = os.environ.get(
    "PANTHEONPLUS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "PP_data"))
PP_DAT = os.path.join(PP_DIR, "Pantheon+SH0ES.dat")
PP_COV = os.path.join(PP_DIR, "Pantheon+SH0ES_STAT+SYS.cov")


def auto_download_pp():
    """Sparse-checkout PP data from github.com/PantheonPlusSH0ES/DataRelease."""
    print(f"\nPantheon+ data not found at {PP_DIR}.")
    print("Attempting automatic download from GitHub...")
    print(f"  Target directory: {PP_DIR}")
    print(f"  Required disk: ~33 MB")
    response = input("\nDownload now? [Y/n] ").strip().lower()
    if response and response != 'y':
        print("Aborted. To resume, either:")
        print("  (a) Set PANTHEONPLUS_DIR to a directory containing the files.")
        print("  (b) Manually clone:")
        print("      git clone --depth=1 --filter=blob:none --no-checkout \\")
        print("          https://github.com/PantheonPlusSH0ES/DataRelease.git")
        print("      cd DataRelease")
        print("      git sparse-checkout init --cone")
        print("      git sparse-checkout set 'Pantheon+_Data/4_DISTANCES_AND_COVAR'")
        print("      git checkout main")
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
                print("ERROR:", r.stderr)
                sys.exit(1)

        src = os.path.join(tmp, 'DataRelease', 'Pantheon+_Data',
                           '4_DISTANCES_AND_COVAR')
        for fname in ['Pantheon+SH0ES.dat', 'Pantheon+SH0ES_STAT+SYS.cov']:
            src_f = os.path.join(src, fname)
            dst_f = os.path.join(PP_DIR, fname)
            print(f"  Copying {fname} ...")
            with open(src_f, 'rb') as fi, open(dst_f, 'wb') as fo:
                fo.write(fi.read())
    print("Download complete.\n")


def load_pantheonplus():
    """Load PP data and full STAT+SYS covariance, return dict."""
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
    # Read covariance
    with open(PP_COV) as f:
        first = f.readline().strip()
    Ncov = int(first)
    cov = np.loadtxt(PP_COV, skiprows=1).reshape((Ncov, Ncov))
    out['Cinv'] = np.linalg.inv(cov)
    return out


# ============================================================================
# SECTION 3: BACKGROUND
# ============================================================================

# Dense z-grid for line-of-sight integrals
_Z_GRID = np.concatenate([[0.0], np.geomspace(1e-3, 5.0, 200)])


def E_lcdm(z, Om):
    """Flat LCDM normalized Hubble."""
    return np.sqrt(Om*(1+z)**3 + (1-Om))


def T_gauss(z, A, zc, w):
    """Locked Gaussian modulation T(z) = 1 + A*exp(-((z-zc)/w)^2)."""
    return 1.0 + A * np.exp(-((z - zc)/w)**2)


def comoving_dist_grid(H0, Om, T_func=None):
    """D_C(z) on _Z_GRID. T_func is None for LCDM, else callable."""
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
# SECTION 4: PANTHEON+ LIKELIHOOD
# ============================================================================

def chi2_panplus(H0, Om, T_func, pp):
    """Pantheon+ unbinned chi^2 with M_B profiled analytically."""
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
# SECTION 5: BAO DATA AND LIKELIHOOD (DR1, DR2, BOSS, eBOSS)
# ============================================================================

# DESI DR2 BAO (DESI Collaboration 2025, arXiv:2503.14738/14739)
# Format: DV = (z, DV/rd, sigma); pair = (z, DM/rd, sDM, DH/rd, sDH, corr)
DESI_DR2_DV   = [(0.295, 7.942, 0.075)]  # BGS
DESI_DR2_PAIR = [
    ('LRG1', 0.510, 13.587, 0.169, 21.863, 0.427, -0.475),
    ('LRG2', 0.706, 17.347, 0.180, 19.458, 0.332, -0.423),
    ('LRG3', 0.934, 21.574, 0.153, 17.641, 0.193, -0.425),
    ('QSO',  1.321, 27.605, 0.320, 14.178, 0.217, -0.437),
    ('ELG2', 1.484, 30.519, 0.758, 12.816, 0.513, -0.489),
    ('Lya',  2.330, 38.988, 0.531,  8.632, 0.101, -0.431),
]

# DESI DR1 BAO: 1 isotropic + 5 anisotropic = 11 measurements
DESI_DR1_DV   = [(0.295, 7.93, 0.150)]
DESI_DR1_PAIR = [
    ('LRG1', 0.510, 13.62, 0.25, 20.98, 0.61, -0.445),
    ('LRG2', 0.706, 16.85, 0.32, 20.08, 0.60, -0.420),
    ('LRG3', 0.930, 21.71, 0.28, 17.88, 0.35, -0.389),
    ('ELG2', 1.317, 27.79, 0.69, 13.82, 0.42, -0.444),
    ('Lya',  2.330, 39.71, 0.94,  8.52, 0.17, -0.477),
]

# BOSS DR12 (Alam+ 2017)
BOSS_DM = [(0.38, 10.27, 0.15)]
BOSS_DH = [(0.38, 24.89, 0.58, -0.42)]

# eBOSS DR16 (Alam+ 2021)
EBOSS_DM_PAIRS = [
    ('eBOSS_LRG', 0.698, 17.86, 0.33, 19.33, 0.53, -0.40),
    ('eBOSS_QSO', 1.480, 30.21, 0.79, 13.23, 0.47, -0.40),
    ('eBOSS_Lya', 2.334, 37.60, 1.90,  8.93, 0.28, -0.45),
]
EBOSS_DV = [('eBOSS_ELG', 0.845, 18.33, 0.57)]


# ============================================================================
# SECTION 6: DISTANCE PRIORS (sound-horizon + multiplicative calibration)
# ============================================================================

def _z_star_HS(Ob_h2, Om_h2):
    """Hu-Sugiyama 1996 fitting formula for z_*."""
    g1 = 0.0783 * Ob_h2**(-0.238) / (1 + 39.5 * Ob_h2**0.763)
    g2 = 0.560 / (1 + 21.1 * Ob_h2**1.81)
    return 1048 * (1 + 0.00124 * Ob_h2**(-0.738)) * (1 + g1 * Om_h2**g2)


def _R_lA_raw(H0, Om, Ob_h2, T_func=None):
    """Raw R, l_A, D_M(z*), r_s(z*) without calibration."""
    h = H0/100.0
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

    DM_star, _ = quad(lambda z: C_KMS/H_E(z), 0, z_star, limit=400)
    rs, _ = quad(
        lambda z: 1.0/np.sqrt(3*(1 + 3*Ob_h2/(4*OMEGA_GAMMA_H2)/(1+z)))
                  * C_KMS/H_E(z),
        z_star, 1e6, limit=400)
    R_raw = np.sqrt(Om_h2) * DM_star * 100.0 / C_KMS
    lA_raw = PI * DM_star / rs
    return R_raw, lA_raw, DM_star, rs, z_star


# Calibration: at Planck fiducial flat LCDM, multiplicative factor
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


# Output directory for JSON results from --baseline, --ka, etc.
OUT_DIR = os.environ.get(
    "LVC_V16_OUT", os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# COMBINED-LIKELIHOOD CHI^2
# ============================================================================
def chi2_bao_combined(H0, Om, T_func, rd, skip=None):
    """DR1 + DR2 + BOSS + eBOSS BAO chi².

    skip : set of label strings.  When a label appears in BOTH DR1 and DR2
           BAO blocks (BGS, LRG1, LRG2 are common labels; the measured
           values differ), it is removed from BOTH releases.  BOSS and
           eBOSS are independent and removed once.

    All four BAO sub-blocks are summed.  BOSS and eBOSS appear once
    (they are not duplicated between releases).
    """
    if skip is None:
        skip = set()

    DC = comoving_dist_grid(H0, Om, T_func)
    if DC is None:
        return 1e10
    if T_func is None:
        H = H0 * E_lcdm(_Z_GRID, Om)
    else:
        H = H0 * E_lcdm(_Z_GRID, Om) * T_func(_Z_GRID)

    chi2 = 0.0

    # ---- DR2 BAO ----
    if 'BGS' not in skip:
        for z, DVo, sig in DESI_DR2_DV:
            DM = np.interp(z, _Z_GRID, DC)
            Hv = np.interp(z, _Z_GRID, H)
            DV = (z * DM**2 * C_KMS / Hv) ** (1/3)
            chi2 += ((DVo - DV / rd) / sig) ** 2
    for label, z, DMo, sDM, DHo, sDH, corr in DESI_DR2_PAIR:
        if label in skip:
            continue
        DMp = np.interp(z, _Z_GRID, DC) / rd
        DHp = C_KMS / np.interp(z, _Z_GRID, H) / rd
        cov = np.array([[sDM**2, corr * sDM * sDH],
                        [corr * sDM * sDH, sDH**2]])
        d = np.array([DMo - DMp, DHo - DHp])
        chi2 += d @ np.linalg.inv(cov) @ d

    # ---- DR1 BAO (separate measurements) ----
    if 'BGS' not in skip:
        for z, DVo, sig in DESI_DR1_DV:
            DM = np.interp(z, _Z_GRID, DC)
            Hv = np.interp(z, _Z_GRID, H)
            DV = (z * DM**2 * C_KMS / Hv) ** (1/3)
            chi2 += ((DVo - DV / rd) / sig) ** 2
    for label, z, DMo, sDM, DHo, sDH, corr in DESI_DR1_PAIR:
        if label in skip:
            continue
        DMp = np.interp(z, _Z_GRID, DC) / rd
        DHp = C_KMS / np.interp(z, _Z_GRID, H) / rd
        cov = np.array([[sDM**2, corr * sDM * sDH],
                        [corr * sDM * sDH, sDH**2]])
        d = np.array([DMo - DMp, DHo - DHp])
        chi2 += d @ np.linalg.inv(cov) @ d

    # ---- BOSS DR12 (once) ----
    if 'BOSS' not in skip:
        for (z, DMo, sDM), (z2, DHo, sDH, corr) in zip(BOSS_DM, BOSS_DH):
            DMp = np.interp(z, _Z_GRID, DC) / rd
            DHp = C_KMS / np.interp(z, _Z_GRID, H) / rd
            cov = np.array([[sDM**2, corr * sDM * sDH],
                            [corr * sDM * sDH, sDH**2]])
            d = np.array([DMo - DMp, DHo - DHp])
            chi2 += d @ np.linalg.inv(cov) @ d

    # ---- eBOSS DR16 (once) ----
    for label, z, DMo, sDM, DHo, sDH, corr in EBOSS_DM_PAIRS:
        if label in skip:
            continue
        DMp = np.interp(z, _Z_GRID, DC) / rd
        DHp = C_KMS / np.interp(z, _Z_GRID, H) / rd
        cov = np.array([[sDM**2, corr * sDM * sDH],
                        [corr * sDM * sDH, sDH**2]])
        d = np.array([DMo - DMp, DHo - DHp])
        chi2 += d @ np.linalg.inv(cov) @ d
    for label, z, DVo, sig in EBOSS_DV:
        if label in skip:
            continue
        DM = np.interp(z, _Z_GRID, DC)
        Hv = np.interp(z, _Z_GRID, H)
        DV = (z * DM**2 * C_KMS / Hv) ** (1/3)
        chi2 += ((DVo - DV / rd) / sig) ** 2

    return float(chi2)


def chi2_total_combined(H0, Om, Ob_h2, T_func, pp, skip=None):
    """PP + (DR1+DR2)BAO + SH0ES + DP, with optional skip set."""
    DM_star = DM_at_zstar(H0, Om, Ob_h2, T_func)
    rd = THETA_STAR * DM_star
    chi_pp, _ = chi2_panplus(H0, Om, T_func, pp)
    chi_bao = chi2_bao_combined(H0, Om, T_func, rd, skip)
    chi_h = ((H0_SHOES - H0) / SIG_H0_SHOES) ** 2
    chi_dp, _, _ = chi2_DP(H0, Om, Ob_h2, T_func)
    return chi_pp + chi_bao + chi_h + chi_dp


# ============================================================================
# FITTERS (combined likelihood)
# ============================================================================
def _de_nm(cost, bounds, seeds=(42, 7), maxiter=120, popsize=12, polish_iter=2500):
    """Differential evolution + Nelder-Mead polish, multi-seed."""
    best = (1e10, None)
    for sd in seeds:
        de = differential_evolution(cost, bounds, seed=sd, tol=1e-7,
                                    maxiter=maxiter, polish=False, popsize=popsize)
        nm = minimize(cost, de.x, method='Nelder-Mead',
                      options=dict(xatol=1e-7, fatol=1e-7, maxiter=polish_iter))
        if nm.fun < best[0]:
            best = (float(nm.fun), nm.x.copy())
    return best


def fit_LCDM_comb(pp, skip=None, seeds=(42, 7)):
    bounds = [(60, 80), (0.20, 0.45), (0.020, 0.025)]
    def cost(p):
        return chi2_total_combined(p[0], p[1], p[2], None, pp, skip)
    return _de_nm(cost, bounds, seeds=seeds)


def fit_locked_comb(pp, zc, w, skip=None, seeds=(42, 7)):
    bounds = [(60, 80), (0.20, 0.45), (0.020, 0.025), (-0.20, 0.20)]
    def cost(p):
        T = lambda z: T_gauss(z, p[3], zc, w)
        return chi2_total_combined(p[0], p[1], p[2], T, pp, skip)
    return _de_nm(cost, bounds, seeds=seeds)


def fit_freeGauss_comb(pp, seeds=(42, 7, 13, 99, 31)):
    bounds = [(60, 80), (0.20, 0.45), (0.020, 0.025),
              (-0.20, 0.20), (0.30, 1.20), (0.05, 0.40)]
    def cost(p):
        T = lambda z: T_gauss(z, p[3], p[4], p[5])
        return chi2_total_combined(p[0], p[1], p[2], T, pp)
    return _de_nm(cost, bounds, seeds=seeds, maxiter=150)


# ============================================================================
# §2-§5.1: BASELINE FITS
# ============================================================================
def baseline_fits(pp):
    print("=" * 78)
    print("§2-§5: combined-fit baselines  (N = %d)" % N_COMBINED)
    print("=" * 78)

    out = {}

    # LCDM
    print("\n[LCDM] fitting ...", flush=True)
    t0 = time.time()
    chi, p = fit_LCDM_comb(pp)
    out['LCDM'] = dict(k=3, chi2=chi, H0=float(p[0]), Om=float(p[1]),
                       Ob=float(p[2]))
    print("  χ² = %.4f   H0=%.3f  Om=%.5f  [%.0fs]"
          % (chi, p[0], p[1], time.time() - t0))

    # LockB, LockC, LockPhi-A, LockPhi-B
    LOCKS = [
        ('LockB',     ZC_LOCKB, W_LOCKBC),
        ('LockC',     ZC_LOCKC, W_LOCKBC),
        ('LockPhi-A', INV_PHI,  W_PHI_A),
        ('LockPhi-B', INV_PHI,  W_PHI_B),
    ]
    for name, zc, w in LOCKS:
        print("\n[%s] zc=%.5f w=%.5f ..." % (name, zc, w), flush=True)
        t0 = time.time()
        chi, p = fit_locked_comb(pp, zc, w)
        out[name] = dict(k=4, zc=zc, w=w, chi2=chi,
                         H0=float(p[0]), Om=float(p[1]), Ob=float(p[2]),
                         A=float(p[3]))
        chi_L = out['LCDM']['chi2']
        dBIC = (chi - chi_L) + np.log(N_COMBINED)
        out[name]['dBIC'] = float(dBIC)
        print("  χ² = %.4f   A=%+.5f   ΔBIC = %+.3f   [%.0fs]"
              % (chi, p[3], dBIC, time.time() - t0))

    # Free Gaussian (5 seeds, slower)
    print("\n[free Gaussian (k=6, 5 seeds)] ...", flush=True)
    t0 = time.time()
    chi, p = fit_freeGauss_comb(pp)
    out['free_Gaussian'] = dict(k=6, chi2=chi,
                                H0=float(p[0]), Om=float(p[1]), Ob=float(p[2]),
                                A=float(p[3]), zc=float(p[4]), w=float(p[5]))
    print("  χ² = %.4f   zc=%.5f  w=%.5f  A=%+.5f   [%.0fs]"
          % (chi, p[4], p[5], p[3], time.time() - t0))

    # Print summary
    print("\n" + "=" * 78)
    print("Summary table (Combined fit, N=%d)" % N_COMBINED)
    print("=" * 78)
    print(f"{'Model':<14}{'k':<3}{'χ²':<13}{'H0':<8}{'Om':<8}{'A':<11}{'ΔBIC'}")
    print("-" * 78)
    print(f"{'LCDM':<14}{out['LCDM']['k']:<3}{out['LCDM']['chi2']:<13.4f}"
          f"{out['LCDM']['H0']:<8.3f}{out['LCDM']['Om']:<8.4f}"
          f"{'-':<11}{'baseline'}")
    for name in ['LockB', 'LockC', 'LockPhi-A', 'LockPhi-B']:
        r = out[name]
        print(f"{name:<14}{r['k']:<3}{r['chi2']:<13.4f}{r['H0']:<8.3f}"
              f"{r['Om']:<8.4f}{r['A']:<+11.5f}{r['dBIC']:<+.3f}")
    fg = out['free_Gaussian']
    print(f"{'free Gauss':<14}{fg['k']:<3}{fg['chi2']:<13.4f}{fg['H0']:<8.3f}"
          f"{fg['Om']:<8.4f}{fg['A']:<+11.5f}{'-'}")
    print(f"  free best (zc, w) = ({fg['zc']:.5f}, {fg['w']:.5f})")
    print(f"  1/φ = {INV_PHI:.5f}  (deviation {100*(fg['zc']-INV_PHI)/INV_PHI:+.3f}%)")
    print(f"  e/(5π) = {W_PHI_A:.5f}  (deviation {100*(fg['w']-W_PHI_A)/W_PHI_A:+.3f}%)")

    save_json(out, 'v16_baseline.json')
    return out


# ============================================================================
# §5.2: z_c LADDER
# ============================================================================
def zc_ladder(pp, w_lock=None):
    """Scan z_c at fixed w = e/(5π).  Single seed for speed."""
    if w_lock is None:
        w_lock = W_PHI_A

    print("\n" + "=" * 78)
    print("§5.2: z_c natural-constant ladder  (w fixed at e/(5π))")
    print("=" * 78)

    LADDER = [
        ('1/φ',          INV_PHI),
        ('0.6200',       0.6200),     # round number, no NC interpretation
        ('0.6150',       0.6150),
        ('5/8',          5.0/8.0),
        ('√(π/8)',       np.sqrt(PI/8)),
        ('11/18',        11.0/18.0),
        ('0.6100',       0.6100),
        ('0.6300',       0.6300),
        ('1/√e',         1.0/np.sqrt(np.e)),
    ]

    results = []
    print(f"\n{'label':<12}{'zc':<10}{'χ²':<13}{'time'}")
    print("-" * 50)
    for label, zc in LADDER:
        t0 = time.time()
        chi, p = fit_locked_comb(pp, zc, w_lock, seeds=(42,))
        elapsed = time.time() - t0
        results.append(dict(label=label, zc=float(zc), chi2=chi,
                            H0=float(p[0]), Om=float(p[1]), Ob=float(p[2]),
                            A=float(p[3]), w=float(w_lock), time=elapsed))
        print(f"{label:<12}{zc:<10.5f}{chi:<13.4f}{elapsed:.0f}s", flush=True)

    # Sort and report
    results.sort(key=lambda r: r['chi2'])
    chi_min = results[0]['chi2']
    print("\nSorted (best first):")
    print(f"{'rank':<5}{'label':<12}{'zc':<10}{'χ²':<13}{'Δχ² vs best'}")
    print("-" * 60)
    for i, r in enumerate(results, 1):
        print(f"{i:<5}{r['label']:<12}{r['zc']:<10.5f}{r['chi2']:<13.4f}"
              f"{r['chi2']-chi_min:+.4f}")

    save_json(results, 'v16_ladder.json')
    return results


# ============================================================================
# §5.3: 2D LANDSCAPE
# ============================================================================
def landscape_2d(pp, chi_LCDM=None):
    """5×5 grid on (z_c, w) around LockPhi center."""
    print("\n" + "=" * 78)
    print("§5.3: 2D χ² landscape on (z_c, w)")
    print("=" * 78)
    print("zc grid: ±5% around 1/φ;  w grid: ±15% around e/(5π)")

    zc_grid = INV_PHI * np.array([0.95, 0.975, 1.0, 1.025, 1.05])
    w_grid  = W_PHI_A * np.array([0.85, 0.925, 1.0, 1.075, 1.15])

    if chi_LCDM is None:
        # Run quick LCDM fit if not provided
        chi_LCDM, _ = fit_LCDM_comb(pp)
    print(f"\nLCDM reference χ² = {chi_LCDM:.4f}")
    print(f"\n{'zc':<10}{'w':<10}{'χ²':<13}{'ΔBIC':<11}{'A':<10}{'time'}")
    print("-" * 70)

    landscape = []
    # Warm-start chain
    x0 = [69.43, 0.286, 0.0228, 0.052]
    penalty = np.log(N_COMBINED)

    for zc in zc_grid:
        for w in w_grid:
            t0 = time.time()
            def cost(p):
                T = lambda z: T_gauss(z, p[3], zc, w)
                return chi2_total_combined(p[0], p[1], p[2], T, pp)
            nm = minimize(cost, x0, method='Nelder-Mead',
                          options=dict(xatol=1e-6, fatol=1e-6, maxiter=1000))
            chi, p = float(nm.fun), nm.x
            x0 = p  # warm start
            elapsed = time.time() - t0
            dBIC = (chi - chi_LCDM) + penalty
            landscape.append(dict(zc=float(zc), w=float(w), chi2=chi,
                                  H0=float(p[0]), Om=float(p[1]),
                                  Ob=float(p[2]), A=float(p[3]),
                                  dBIC=float(dBIC), time=elapsed))
            print(f"{zc:<10.5f}{w:<10.5f}{chi:<13.4f}{dBIC:<+11.3f}"
                  f"{p[3]:<+10.5f}{elapsed:.0f}s", flush=True)

    # Reshape and print as ΔBIC table
    print("\nΔBIC table (rows=zc, cols=w):")
    print(f"{'zc \\ w':<10}", end='')
    for w in w_grid:
        print(f"{w:<11.4f}", end='')
    print()
    print("-" * 70)
    for zc in zc_grid:
        print(f"{zc:<10.5f}", end='')
        for w in w_grid:
            r = next(x for x in landscape
                     if abs(x['zc']-zc) < 1e-9 and abs(x['w']-w) < 1e-9)
            print(f"{r['dBIC']:<+11.3f}", end='')
        print()

    save_json(dict(zc_grid=zc_grid.tolist(), w_grid=w_grid.tolist(),
                   chi_LCDM=float(chi_LCDM), landscape=landscape),
              'v16_landscape.json')
    return landscape


# ============================================================================
# §6.1: K-A MULTI-LOO ON COMBINED LIKELIHOOD
# ============================================================================
KA_COMBOS = [
    ('LRG1',),
    ('LRG2',),
    ('BOSS',),
    ('eBOSS_LRG',),
    ('LRG1', 'LRG2'),
    ('LRG1', 'eBOSS_LRG'),
    ('LRG1', 'BOSS'),
    ('LRG1', 'LRG2', 'eBOSS_LRG'),
    ('LRG1', 'LRG2', 'BOSS'),
    ('LRG1', 'BOSS', 'eBOSS_LRG'),
    ('LRG1', 'LRG2', 'BOSS', 'eBOSS_LRG'),
]

# Combined-LOO measurement counts.  LRG1, LRG2 appear in BOTH releases,
# so removing them drops 2+2 = 4 measurements each.  BOSS, eBOSS_LRG
# appear once each.
LABEL_DELTA_N_COMBINED = {
    'LRG1': 4,
    'LRG2': 4,
    'BOSS': 2,
    'eBOSS_LRG': 2,
}


def ka_combined(pp, zc=None, w=None):
    if zc is None:
        zc = INV_PHI
    if w is None:
        w = W_PHI_A
    print("\n" + "=" * 78)
    print(f"§6.1: K-A multi-LOO on combined BAO  (zc={zc:.5f}, w={w:.5f})")
    print("=" * 78)

    out = {}
    print(f"\n{'Subset removed':<35}{'dN':<5}{'χ²(L)':<11}{'χ²(lock)':<12}"
          f"{'Δχ²':<11}{'ΔBIC':<10}{'verdict'}")
    print("-" * 80)

    for combo in KA_COMBOS:
        key = '+'.join(combo)
        skip = set(combo)
        dN = sum(LABEL_DELTA_N_COMBINED.get(lbl, 0) for lbl in combo)
        N_eff = N_COMBINED - dN

        t0 = time.time()
        chi_L, _ = fit_LCDM_comb(pp, skip=skip, seeds=(42,))
        chi_lk, p_lk = fit_locked_comb(pp, zc, w, skip=skip, seeds=(42,))
        dchi = chi_lk - chi_L
        dBIC = dchi + np.log(N_eff)  # +1 parameter for lock
        verdict = ('WIN' if dBIC < -2 else 'tie' if dBIC < 2 else 'BURN')

        out[key] = dict(combo=list(combo), dN=dN, N_eff=int(N_eff),
                        chi2_L=float(chi_L), chi2_lock=float(chi_lk),
                        H0=float(p_lk[0]), Om=float(p_lk[1]),
                        Ob=float(p_lk[2]), A=float(p_lk[3]),
                        dchi=float(dchi), dBIC=float(dBIC),
                        verdict=verdict, sec=time.time()-t0)

        print(f"{key:<35}{dN:<5}{chi_L:<11.3f}{chi_lk:<12.3f}"
              f"{dchi:<+11.3f}{dBIC:<+10.3f}{verdict}", flush=True)

    n_win = sum(1 for r in out.values() if r['verdict'] == 'WIN')
    n_tie = sum(1 for r in out.values() if r['verdict'] == 'tie')
    n_burn = sum(1 for r in out.values() if r['verdict'] == 'BURN')
    print(f"\nVerdict count: {n_win} WIN, {n_tie} tie, {n_burn} BURN out of {len(out)}")

    save_json(out, 'v16_ka_combined.json')
    return out


# ============================================================================
# §6.2: FREE-Ω_Λ TEST
# ============================================================================
# Re-implement curved-OL machinery here (compact form)


def E_curved(z, Om, OL, Or=0.0):
    Ok = 1.0 - Om - OL - Or
    return np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + Ok*(1+z)**2 + OL)


def comoving_dist_grid_curved(H0, Om, OL, T_func=None):
    if T_func is None:
        H = H0 * E_curved(_Z_GRID, Om, OL)
    else:
        Tv = T_func(_Z_GRID)
        if np.min(Tv) < 0.4 or np.max(Tv) > 2.0:
            return None
        H = H0 * E_curved(_Z_GRID, Om, OL) * Tv
    if np.any(H <= 0):
        return None
    integ = C_KMS / H
    DC = np.zeros_like(_Z_GRID)
    DC[1:] = np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(_Z_GRID))
    return DC


def chi2_panplus_curved(H0, Om, OL, T_func, pp):
    DC = comoving_dist_grid_curved(H0, Om, OL, T_func)
    if DC is None:
        return 1e10, 0.0
    DM = np.interp(pp['zHD'], _Z_GRID, DC)
    mu_pred = np.where(pp['is_calib'], pp['ceph'],
                       5*np.log10((1+pp['zHEL']) * DM) + 25)
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


def chi2_bao_curved_combined(H0, Om, OL, T_func, rd):
    DC = comoving_dist_grid_curved(H0, Om, OL, T_func)
    if DC is None:
        return 1e10
    if T_func is None:
        H = H0 * E_curved(_Z_GRID, Om, OL)
    else:
        H = H0 * E_curved(_Z_GRID, Om, OL) * T_func(_Z_GRID)
    chi2 = 0.0
    # Use the same loops as flat case but with curved DC, H
    for z, DVo, sig in DESI_DR2_DV:
        DM = np.interp(z, _Z_GRID, DC)
        Hv = np.interp(z, _Z_GRID, H)
        DV = (z*DM**2*C_KMS/Hv)**(1/3)
        chi2 += ((DVo - DV/rd)/sig)**2
    for label, z, DMo, sDM, DHo, sDH, corr in DESI_DR2_PAIR:
        DMp = np.interp(z, _Z_GRID, DC)/rd
        DHp = C_KMS/np.interp(z, _Z_GRID, H)/rd
        cov = np.array([[sDM**2, corr*sDM*sDH],[corr*sDM*sDH, sDH**2]])
        d = np.array([DMo-DMp, DHo-DHp])
        chi2 += d @ np.linalg.inv(cov) @ d
    for z, DVo, sig in DESI_DR1_DV:
        DM = np.interp(z, _Z_GRID, DC)
        Hv = np.interp(z, _Z_GRID, H)
        DV = (z*DM**2*C_KMS/Hv)**(1/3)
        chi2 += ((DVo - DV/rd)/sig)**2
    for label, z, DMo, sDM, DHo, sDH, corr in DESI_DR1_PAIR:
        DMp = np.interp(z, _Z_GRID, DC)/rd
        DHp = C_KMS/np.interp(z, _Z_GRID, H)/rd
        cov = np.array([[sDM**2, corr*sDM*sDH],[corr*sDM*sDH, sDH**2]])
        d = np.array([DMo-DMp, DHo-DHp])
        chi2 += d @ np.linalg.inv(cov) @ d
    for (z, DMo, sDM), (z2, DHo, sDH, corr) in zip(BOSS_DM, BOSS_DH):
        DMp = np.interp(z, _Z_GRID, DC)/rd
        DHp = C_KMS/np.interp(z, _Z_GRID, H)/rd
        cov = np.array([[sDM**2, corr*sDM*sDH],[corr*sDM*sDH, sDH**2]])
        d = np.array([DMo-DMp, DHo-DHp])
        chi2 += d @ np.linalg.inv(cov) @ d
    for label, z, DMo, sDM, DHo, sDH, corr in EBOSS_DM_PAIRS:
        DMp = np.interp(z, _Z_GRID, DC)/rd
        DHp = C_KMS/np.interp(z, _Z_GRID, H)/rd
        cov = np.array([[sDM**2, corr*sDM*sDH],[corr*sDM*sDH, sDH**2]])
        d = np.array([DMo-DMp, DHo-DHp])
        chi2 += d @ np.linalg.inv(cov) @ d
    for label, z, DVo, sig in EBOSS_DV:
        DM = np.interp(z, _Z_GRID, DC)
        Hv = np.interp(z, _Z_GRID, H)
        DV = (z*DM**2*C_KMS/Hv)**(1/3)
        chi2 += ((DVo - DV/rd)/sig)**2
    return float(chi2)


def _R_lA_curved(H0, Om, OL, Ob_h2, T_func=None):
    h = H0/100.0
    Om_h2 = Om * h * h
    z_star = _z_star_HS(Ob_h2, Om_h2)
    Or_frac = OMEGA_R_H2 / h**2
    Ok_frac = 1.0 - Om - Or_frac - OL
    if T_func is None:
        def H_E(z):
            return H0*np.sqrt(Om*(1+z)**3 + Or_frac*(1+z)**4
                              + Ok_frac*(1+z)**2 + OL)
    else:
        def H_E(z):
            T_val = float(T_func(np.array([z]))[0])
            return H0*np.sqrt(Om*(1+z)**3 + Or_frac*(1+z)**4
                              + Ok_frac*(1+z)**2 + OL*T_val*T_val)
    DM_star, _ = quad(lambda z: C_KMS/H_E(z), 0, z_star, limit=400)
    rs, _ = quad(
        lambda z: 1.0/np.sqrt(3*(1 + 3*Ob_h2/(4*OMEGA_GAMMA_H2)/(1+z)))
                  * C_KMS/H_E(z),
        z_star, 1e6, limit=400)
    R_raw = np.sqrt(Om_h2) * DM_star * 100.0 / C_KMS
    lA_raw = PI * DM_star / rs
    return R_raw, lA_raw, DM_star, rs, z_star


# Calibration: same form as v14 (multiplicative)
_R_AT_PLANCK_C, _LA_AT_PLANCK_C, _, _, _ = _R_lA_curved(
    67.36, 0.3153, 1 - 0.3153 - OMEGA_R_H2/(67.36/100)**2, 0.02237, None)
CAL_R_C = R_PLANCK / _R_AT_PLANCK_C
CAL_LA_C = LA_PLANCK / _LA_AT_PLANCK_C


def chi2_DP_curved(H0, Om, OL, Ob_h2, T_func=None):
    R_raw, lA_raw, _, _, _ = _R_lA_curved(H0, Om, OL, Ob_h2, T_func)
    R_cal = R_raw * CAL_R_C
    lA_cal = lA_raw * CAL_LA_C
    x = np.array([R_cal, lA_cal, Ob_h2])
    xref = np.array([R_PLANCK, LA_PLANCK, WB_PLANCK])
    d = x - xref
    return float(d @ COVINV_DP @ d)


def DM_star_curved(H0, Om, OL, Ob_h2, T_func=None):
    _, _, DM_star, _, _ = _R_lA_curved(H0, Om, OL, Ob_h2, T_func)
    return DM_star


def chi2_total_curved_comb(H0, Om, OL, Ob_h2, T_func, pp):
    DM_star = DM_star_curved(H0, Om, OL, Ob_h2, T_func)
    rd = THETA_STAR * DM_star
    chi_pp, _ = chi2_panplus_curved(H0, Om, OL, T_func, pp)
    chi_bao = chi2_bao_curved_combined(H0, Om, OL, T_func, rd)
    chi_h = ((H0_SHOES - H0) / SIG_H0_SHOES)**2
    chi_dp = chi2_DP_curved(H0, Om, OL, Ob_h2, T_func)
    return chi_pp + chi_bao + chi_h + chi_dp


def fit_LCDM_curved_comb(pp):
    bounds = [(60, 80), (0.20, 0.45), (0.50, 0.85), (0.020, 0.025)]
    def cost(p):
        return chi2_total_curved_comb(p[0], p[1], p[2], p[3], None, pp)
    return _de_nm(cost, bounds, seeds=(42, 7), maxiter=120)


def fit_LockPhi_curved_comb(pp, zc=None, w=None):
    if zc is None: zc = INV_PHI
    if w is None: w = W_PHI_A
    bounds = [(60, 80), (0.20, 0.45), (0.50, 0.85),
              (0.020, 0.025), (-0.20, 0.20)]
    def cost(p):
        T = lambda z: T_gauss(z, p[4], zc, w)
        return chi2_total_curved_comb(p[0], p[1], p[2], p[3], T, pp)
    return _de_nm(cost, bounds, seeds=(42,), maxiter=80, popsize=10)


def free_OL_test(pp, baseline=None):
    print("\n" + "=" * 78)
    print("§6.2: Free-Ω_Λ test on combined likelihood")
    print("=" * 78)

    if baseline is None or 'LCDM' not in baseline or 'LockPhi-A' not in baseline:
        print("(Need baseline fits.  Running LCDM and LockPhi-A first ...)")
        chi_flatL, _ = fit_LCDM_comb(pp)
        chi_flatP, p_flatP = fit_locked_comb(pp, INV_PHI, W_PHI_A)
        flatA = float(p_flatP[3])
    else:
        chi_flatL = baseline['LCDM']['chi2']
        chi_flatP = baseline['LockPhi-A']['chi2']
        flatA = baseline['LockPhi-A']['A']

    print(f"\n[free-OL LCDM (kCDM, k=4)] ...", flush=True)
    t0 = time.time()
    chi_LF, p_LF = fit_LCDM_curved_comb(pp)
    print(f"  χ² = {chi_LF:.4f}  H0={p_LF[0]:.3f}  Om={p_LF[1]:.4f}  "
          f"OL={p_LF[2]:.4f}  Ok={1-p_LF[1]-p_LF[2]:+.4f}  [{time.time()-t0:.0f}s]")

    print(f"\n[free-OL LockPhi (k=5)] ...", flush=True)
    t0 = time.time()
    chi_PF, p_PF = fit_LockPhi_curved_comb(pp)
    print(f"  χ² = {chi_PF:.4f}  H0={p_PF[0]:.3f}  Om={p_PF[1]:.4f}  "
          f"OL={p_PF[2]:.4f}  Ok={1-p_PF[1]-p_PF[2]:+.4f}  A={p_PF[4]:+.5f}  "
          f"[{time.time()-t0:.0f}s]")

    # Summary
    print("\nSummary:")
    print(f"{'Model':<26}{'k':<3}{'χ²':<12}{'A':<10}{'ΔBIC vs flat-LCDM'}")
    print("-" * 78)
    print(f"{'flat LCDM':<26}{'3':<3}{chi_flatL:<12.4f}{'-':<10}{'baseline'}")
    bic_flatP = chi_flatP - chi_flatL + np.log(N_COMBINED)*1
    print(f"{'flat LockPhi':<26}{'4':<3}{chi_flatP:<12.4f}{flatA:<+10.5f}{bic_flatP:<+.3f}")
    bic_kCDM = chi_LF - chi_flatL + np.log(N_COMBINED)*1
    print(f"{'kCDM':<26}{'4':<3}{chi_LF:<12.4f}{'-':<10}{bic_kCDM:<+.3f}")
    bic_curvP = chi_PF - chi_flatL + np.log(N_COMBINED)*2
    print(f"{'free-OL LockPhi':<26}{'5':<3}{chi_PF:<12.4f}{p_PF[4]:<+10.5f}{bic_curvP:<+.3f}")

    bic_phi_vs_kcdm = chi_PF - chi_LF + np.log(N_COMBINED)
    print(f"\nLockPhi vs kCDM ΔBIC = {bic_phi_vs_kcdm:+.3f}")
    A_drift = float(p_PF[4]) - flatA
    print(f"A drift (free-OL vs flat) = {A_drift:+.5f}  ({100*A_drift/flatA:+.1f}%)")

    out = dict(
        flat_LCDM_chi2=chi_flatL, flat_LockPhi_chi2=chi_flatP, flat_A=flatA,
        kCDM=dict(chi2=float(chi_LF), H0=float(p_LF[0]), Om=float(p_LF[1]),
                  OL=float(p_LF[2]), Ob=float(p_LF[3]),
                  Ok=float(1-p_LF[1]-p_LF[2])),
        free_OL_LockPhi=dict(chi2=float(chi_PF), H0=float(p_PF[0]),
                             Om=float(p_PF[1]), OL=float(p_PF[2]),
                             Ob=float(p_PF[3]), A=float(p_PF[4]),
                             Ok=float(1-p_PF[1]-p_PF[2])),
        dBIC_LockPhi_vs_kCDM=float(bic_phi_vs_kcdm),
        A_drift=float(A_drift),
    )
    save_json(out, 'v16_freeOL.json')
    return out


# ============================================================================
# §6.3: fσ8 JOINT
# ============================================================================
def fsigma8_joint(pp, baseline=None):
    print("\n" + "=" * 78)
    print("§6.3: fσ8 joint analysis (19-pt v11 §3 compilation)")
    print("=" * 78)

    if baseline is None:
        print("ERROR: need baseline fits.  Run --baseline first.")
        return None

    # 19-point fσ8 dataset (v11 §3)
    FSIGMA8 = [
        (0.067, 0.423, 0.055), (0.150, 0.490, 0.145),
        (0.180, 0.360, 0.090), (0.380, 0.440, 0.060),
        (0.32,  0.473, 0.041), (0.57,  0.467, 0.045),
        (0.44,  0.413, 0.080), (0.60,  0.390, 0.063),
        (0.73,  0.437, 0.072), (0.60,  0.55,  0.12 ),
        (0.86,  0.40,  0.11 ), (0.698, 0.473, 0.044),
        (0.85,  0.315, 0.095), (1.48,  0.462, 0.045),
        (2.334, 0.402, 0.099), (0.510, 0.450, 0.040),
        (0.706, 0.470, 0.045), (0.930, 0.435, 0.045),
        (1.317, 0.388, 0.055),
    ]
    z_arr = np.array([d[0] for d in FSIGMA8])
    obs   = np.array([d[1] for d in FSIGMA8])
    err   = np.array([d[2] for d in FSIGMA8])

    # Use the v15 linear_growth function

    def E_lock(z, Om, A=0, zc=0.5, w=0.1):
        return E_lcdm(z, Om) * T_gauss(z, A, zc, w)

    def Om_at_a(a, Om0, A, zc, w, lcdm=False):
        z = 1.0/a - 1.0
        E = E_lcdm(z, Om0) if lcdm else E_lock(z, Om0, A, zc, w)
        return Om0 * (1+z)**3 / E**2

    def dlnE_dlna(a, Om0, A, zc, w, lcdm=False, eps=1e-5):
        a_p = a*np.exp(eps); a_m = a*np.exp(-eps)
        z_p = 1.0/a_p - 1.0; z_m = 1.0/a_m - 1.0
        if lcdm:
            Ep = E_lcdm(z_p, Om0); Em = E_lcdm(z_m, Om0)
        else:
            Ep = E_lock(z_p, Om0, A, zc, w); Em = E_lock(z_m, Om0, A, zc, w)
        return (np.log(Ep) - np.log(Em)) / (2*eps)

    def growth(z_eval, Om, A=0, zc=0.5, w=0.1, lcdm=False):
        a_init = 1e-3
        def rhs(lna, y):
            D, dD = y
            a = np.exp(lna)
            Oma  = Om_at_a(a, Om, A, zc, w, lcdm)
            dlnE = dlnE_dlna(a, Om, A, zc, w, lcdm)
            ddD = -(2 + dlnE)*dD + 1.5*Oma*D
            return [dD, ddD]
        z_ext = sorted(set(list(z_eval) + [0.0]))
        a_q = sorted([1.0/(1+zz) for zz in z_ext if 1.0/(1+zz) > a_init])
        lna_q = [np.log(aq) for aq in a_q]
        sol = solve_ivp(rhs, [np.log(a_init), 0.0], [a_init, a_init],
                        t_eval=lna_q, method='RK45',
                        rtol=1e-8, atol=1e-10, max_step=0.05)
        if not sol.success: return None, None
        D_grid = sol.y[0]; f_grid = sol.y[1] / sol.y[0]
        idx0 = np.argmin(np.abs(np.exp(sol.t) - 1.0))
        D0 = D_grid[idx0]
        a_eval = np.array([1.0/(1+zz) for zz in z_eval])
        f_e = np.interp(np.log(a_eval), sol.t, f_grid)
        D_e = np.interp(np.log(a_eval), sol.t, D_grid) / D0
        return f_e, D_e

    def chi2_fs8(Om, s8, A=0, zc=0.5, w=0.1, lcdm=False):
        f_e, D_e = growth(z_arr, Om, A, zc, w, lcdm)
        if f_e is None: return 1e10
        pred = f_e * s8 * D_e
        return float(np.sum(((obs - pred) / err)**2))

    def fit_s8(Om, A=0, zc=0.5, w=0.1, lcdm=False):
        res = minimize_scalar(lambda s8: chi2_fs8(Om, s8, A, zc, w, lcdm),
                              bounds=(0.5, 1.2), method='bounded',
                              options=dict(xatol=1e-5))
        return res.x, res.fun

    out = {}
    print(f"\n{'Model':<14}{'Om':<10}{'σ8(0)':<10}{'χ²(fσ8)':<11}{'joint ΔBIC'}")
    print("-" * 78)
    chi_geom_L = baseline['LCDM']['chi2']
    Om_L = baseline['LCDM']['Om']
    s8_L, chi_fL = fit_s8(Om_L, lcdm=True)
    total_L = chi_geom_L + chi_fL
    out['LCDM'] = dict(Om=Om_L, sigma8=float(s8_L), chi2_fs8=float(chi_fL),
                       joint_total=float(total_L))
    print(f"{'LCDM':<14}{Om_L:<10.5f}{s8_L:<10.4f}{chi_fL:<11.4f}{'baseline'}")

    Njoint = N_COMBINED + 19
    for name, key in [('LockB','LockB'), ('LockC','LockC'),
                      ('LockPhi','LockPhi-A')]:
        b = baseline[key]
        s8, chi_f = fit_s8(b['Om'], A=b['A'], zc=b['zc'], w=b['w'], lcdm=False)
        total = b['chi2'] + chi_f
        dBIC = (total - total_L) + np.log(Njoint)
        out[name] = dict(Om=b['Om'], A=b['A'], zc=b['zc'], w=b['w'],
                         sigma8=float(s8), chi2_fs8=float(chi_f),
                         joint_total=float(total), joint_dBIC=float(dBIC))
        print(f"{name:<14}{b['Om']:<10.5f}{s8:<10.4f}{chi_f:<11.4f}{dBIC:<+.3f}")

    save_json(out, 'v16_fsigma8.json')
    return out


# ============================================================================
# §6.4: BBN STRICT r_d
# ============================================================================
RD_BBN = 147.05
RD_BBN_SIG = 0.3


def chi2_total_bbn(H0, Om, Ob_h2, T_func, pp):
    DM_star = DM_at_zstar(H0, Om, Ob_h2, T_func)
    rd = THETA_STAR * DM_star
    chi_base = chi2_total_combined(H0, Om, Ob_h2, T_func, pp)
    chi_rd = ((rd - RD_BBN) / RD_BBN_SIG)**2
    return chi_base + chi_rd, float(rd)


def fit_LCDM_bbn(pp):
    bounds = [(60, 80), (0.20, 0.45), (0.020, 0.025)]
    def cost(p):
        c, _ = chi2_total_bbn(p[0], p[1], p[2], None, pp)
        return c
    return _de_nm(cost, bounds, seeds=(42, 7))


def fit_LockPhi_bbn(pp, zc=None, w=None):
    if zc is None: zc = INV_PHI
    if w is None: w = W_PHI_A
    bounds = [(60, 80), (0.20, 0.45), (0.020, 0.025), (-0.20, 0.20)]
    def cost(p):
        T = lambda z: T_gauss(z, p[3], zc, w)
        c, _ = chi2_total_bbn(p[0], p[1], p[2], T, pp)
        return c
    return _de_nm(cost, bounds, seeds=(42, 7))


def bbn_test(pp, baseline=None):
    print("\n" + "=" * 78)
    print(f"§6.4: BBN strict r_d = {RD_BBN} ± {RD_BBN_SIG} Mpc")
    print("=" * 78)

    if baseline is None:
        print("(Need baseline.)")
        return None

    print("\n[LCDM with BBN prior] ...", flush=True)
    t0 = time.time()
    chi_L, p_L = fit_LCDM_bbn(pp)
    _, rd_L = chi2_total_bbn(p_L[0], p_L[1], p_L[2], None, pp)
    print(f"  χ² = {chi_L:.4f}  rd = {rd_L:.3f}  [{time.time()-t0:.0f}s]")

    print("\n[LockPhi with BBN prior] ...", flush=True)
    t0 = time.time()
    chi_P, p_P = fit_LockPhi_bbn(pp)
    T = lambda z: T_gauss(z, p_P[3], INV_PHI, W_PHI_A)
    _, rd_P = chi2_total_bbn(p_P[0], p_P[1], p_P[2], T, pp)
    print(f"  χ² = {chi_P:.4f}  A = {p_P[3]:+.5f}  rd = {rd_P:.3f}  "
          f"[{time.time()-t0:.0f}s]")

    N_eff = N_COMBINED + 1
    dBIC_BBN = (chi_P - chi_L) + np.log(N_eff)
    dBIC_unforced = baseline['LockPhi-A']['dBIC']
    shift = dBIC_BBN - dBIC_unforced
    print(f"\nLockPhi ΔBIC unforced = {dBIC_unforced:+.3f}")
    print(f"LockPhi ΔBIC BBN-forced = {dBIC_BBN:+.3f}")
    print(f"Shift = {shift:+.3f}")

    out = dict(
        LCDM_BBN=dict(chi2=float(chi_L), H0=float(p_L[0]), Om=float(p_L[1]),
                      Ob=float(p_L[2]), rd=float(rd_L)),
        LockPhi_BBN=dict(chi2=float(chi_P), H0=float(p_P[0]), Om=float(p_P[1]),
                         Ob=float(p_P[2]), A=float(p_P[3]), rd=float(rd_P)),
        dBIC_unforced=float(dBIC_unforced),
        dBIC_BBN=float(dBIC_BBN), shift=float(shift),
    )
    save_json(out, 'v16_bbn.json')
    return out


# ============================================================================
# §6.5: χ² DECOMPOSITION + BAO POINT-BY-POINT + A-PROFILE
# ============================================================================
def chi2_decomp(pp, baseline):
    """χ² split by data block at LCDM and LockPhi best."""
    print("\n" + "=" * 78)
    print("§6.5a: χ² decomposition by data block")
    print("=" * 78)

    def all_blocks(H0, Om, Ob, T_func):
        DM_star = DM_at_zstar(H0, Om, Ob, T_func)
        rd = THETA_STAR * DM_star
        chi_pp, _ = chi2_panplus(H0, Om, T_func, pp)
        # split BAO
        DC = comoving_dist_grid(H0, Om, T_func)
        if T_func is None:
            H = H0 * E_lcdm(_Z_GRID, Om)
        else:
            H = H0 * E_lcdm(_Z_GRID, Om) * T_func(_Z_GRID)
        # DR2
        chi_dr2 = 0.0
        for z, DVo, sig in DESI_DR2_DV:
            DM = np.interp(z, _Z_GRID, DC); Hv = np.interp(z, _Z_GRID, H)
            DV = (z*DM**2*C_KMS/Hv)**(1/3)
            chi_dr2 += ((DVo - DV/rd)/sig)**2
        for label, z, DMo, sDM, DHo, sDH, corr in DESI_DR2_PAIR:
            DMp = np.interp(z, _Z_GRID, DC)/rd
            DHp = C_KMS/np.interp(z, _Z_GRID, H)/rd
            cov = np.array([[sDM**2, corr*sDM*sDH],[corr*sDM*sDH, sDH**2]])
            d = np.array([DMo-DMp, DHo-DHp])
            chi_dr2 += d @ np.linalg.inv(cov) @ d
        # DR1
        chi_dr1 = 0.0
        for z, DVo, sig in DESI_DR1_DV:
            DM = np.interp(z, _Z_GRID, DC); Hv = np.interp(z, _Z_GRID, H)
            DV = (z*DM**2*C_KMS/Hv)**(1/3)
            chi_dr1 += ((DVo - DV/rd)/sig)**2
        for label, z, DMo, sDM, DHo, sDH, corr in DESI_DR1_PAIR:
            DMp = np.interp(z, _Z_GRID, DC)/rd
            DHp = C_KMS/np.interp(z, _Z_GRID, H)/rd
            cov = np.array([[sDM**2, corr*sDM*sDH],[corr*sDM*sDH, sDH**2]])
            d = np.array([DMo-DMp, DHo-DHp])
            chi_dr1 += d @ np.linalg.inv(cov) @ d
        # BOSS
        chi_boss = 0.0
        for (z, DMo, sDM), (z2, DHo, sDH, corr) in zip(BOSS_DM, BOSS_DH):
            DMp = np.interp(z, _Z_GRID, DC)/rd
            DHp = C_KMS/np.interp(z, _Z_GRID, H)/rd
            cov = np.array([[sDM**2, corr*sDM*sDH],[corr*sDM*sDH, sDH**2]])
            d = np.array([DMo-DMp, DHo-DHp])
            chi_boss += d @ np.linalg.inv(cov) @ d
        # eBOSS
        chi_eboss = 0.0
        for label, z, DMo, sDM, DHo, sDH, corr in EBOSS_DM_PAIRS:
            DMp = np.interp(z, _Z_GRID, DC)/rd
            DHp = C_KMS/np.interp(z, _Z_GRID, H)/rd
            cov = np.array([[sDM**2, corr*sDM*sDH],[corr*sDM*sDH, sDH**2]])
            d = np.array([DMo-DMp, DHo-DHp])
            chi_eboss += d @ np.linalg.inv(cov) @ d
        for label, z, DVo, sig in EBOSS_DV:
            DM = np.interp(z, _Z_GRID, DC); Hv = np.interp(z, _Z_GRID, H)
            DV = (z*DM**2*C_KMS/Hv)**(1/3)
            chi_eboss += ((DVo - DV/rd)/sig)**2
        chi_h = ((H0_SHOES - H0) / SIG_H0_SHOES)**2
        chi_dp, _, _ = chi2_DP(H0, Om, Ob, T_func)
        return dict(pp=float(chi_pp), dr2=float(chi_dr2), dr1=float(chi_dr1),
                    boss=float(chi_boss), eboss=float(chi_eboss),
                    shoes=float(chi_h), dp=float(chi_dp), rd=float(rd))

    L = baseline['LCDM']
    P = baseline['LockPhi-A']
    chi_L = all_blocks(L['H0'], L['Om'], L['Ob'], None)
    T = lambda z: T_gauss(z, P['A'], P['zc'], P['w'])
    chi_P = all_blocks(P['H0'], P['Om'], P['Ob'], T)
    print(f"\n{'Block':<14}{'LCDM χ²':<12}{'LockPhi χ²':<14}{'Δχ²':<12}{'note'}")
    print("-" * 78)
    notes = {'pp':'1701 SN', 'dr2':'13 DESI DR2', 'dr1':'11 DESI DR1',
             'boss':'2 BOSS', 'eboss':'7 eBOSS', 'dp':'3 Planck DP',
             'shoes':'1 SH0ES'}
    for b in ['pp','dr2','dr1','boss','eboss','dp','shoes']:
        d = chi_P[b] - chi_L[b]
        print(f"{b:<14}{chi_L[b]:<12.3f}{chi_P[b]:<14.3f}{d:<+12.3f}{notes[b]}")
    total_L = sum(chi_L[b] for b in ['pp','dr2','dr1','boss','eboss','dp','shoes'])
    total_P = sum(chi_P[b] for b in ['pp','dr2','dr1','boss','eboss','dp','shoes'])
    print("-" * 78)
    print(f"{'Total':<14}{total_L:<12.3f}{total_P:<14.3f}{total_P-total_L:<+12.3f}")
    print(f"\nrd: LCDM = {chi_L['rd']:.3f},  LockPhi = {chi_P['rd']:.3f}")

    out = dict(LCDM=chi_L, LockPhi=chi_P)
    save_json(out, 'v16_decomp.json')
    return out


def Aprofile_LRG12(pp):
    """A-grid scan with LRG1+LRG2 removed from both releases."""
    print("\n" + "=" * 78)
    print("§6.5b: A-profile under LRG1+LRG2 removal")
    print("=" * 78)

    skip = {'LRG1', 'LRG2'}
    A_grid = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08,
              -0.02, -0.05, -0.08]
    print(f"\n{'A':<10}{'χ²':<14}{'H0':<10}{'Om':<10}")
    print("-" * 50)
    out = []
    x0 = [70.0, 0.28, 0.0228]
    for A in A_grid:
        def cost(p):
            T = lambda z: T_gauss(z, A, INV_PHI, W_PHI_A)
            return chi2_total_combined(p[0], p[1], p[2], T, pp, skip)
        nm = minimize(cost, x0, method='Nelder-Mead',
                      options=dict(xatol=1e-6, fatol=1e-6, maxiter=600))
        chi, p = float(nm.fun), nm.x
        x0 = p
        out.append(dict(A=float(A), chi2=chi,
                        H0=float(p[0]), Om=float(p[1]), Ob=float(p[2])))
        print(f"{A:<+10.4f}{chi:<14.4f}{p[0]:<10.3f}{p[1]:<10.5f}", flush=True)
    chi_arr = [r['chi2'] for r in out]
    imin = chi_arr.index(min(chi_arr))
    print(f"\nMinimum at A = {out[imin]['A']:+.4f}, χ² = {out[imin]['chi2']:.4f}")
    print(f"LCDM (A=0) χ² = {out[0]['chi2']:.4f}")
    print(f"Δχ²(best - LCDM) = {out[imin]['chi2'] - out[0]['chi2']:+.4f}")

    save_json(out, 'v16_Aprofile.json')
    return out


# ============================================================================
# §7: DR3 PRIOR PREDICTIONS
# ============================================================================
def dr3_predictions(baseline):
    print("\n" + "=" * 78)
    print("§7: DR3 prior predictions for LockPhi")
    print("=" * 78)

    if baseline is None:
        print("(Need baseline.)")
        return None

    L = baseline['LCDM']
    P = baseline['LockPhi-A']

    def rd_at(H0, Om, Ob, A=0, zc=None, w=None):
        if A == 0:
            T = None
        else:
            T = lambda z: T_gauss(z, A, zc, w)
        DM_star = DM_at_zstar(H0, Om, Ob, T)
        return THETA_STAR * DM_star

    rd_L = rd_at(L['H0'], L['Om'], L['Ob'])
    rd_P = rd_at(P['H0'], P['Om'], P['Ob'], P['A'], P['zc'], P['w'])

    def predict(H0, Om, A, zc_T, w_T, z, rd):
        if A == 0:
            DC = comoving_dist_grid(H0, Om, None)
            H_z = H0 * E_lcdm(np.array([z]), Om)[0]
        else:
            T = lambda zz: T_gauss(zz, A, zc_T, w_T)
            DC = comoving_dist_grid(H0, Om, T)
            H_z = H0 * E_lcdm(np.array([z]), Om)[0] * \
                  T_gauss(np.array([z]), A, zc_T, w_T)[0]
        DM_z = np.interp(z, _Z_GRID, DC)
        return DM_z / rd, C_KMS / H_z / rd

    print(f"\nrd: LCDM = {rd_L:.3f},  LockPhi = {rd_P:.3f}")
    print(f"\n{'z':<8}{'LCDM DM/rd':<14}{'LockPhi DM/rd':<16}{'Δ%':<8}"
          f"{'LCDM DH/rd':<14}{'LockPhi DH/rd':<16}{'Δ%'}")
    print("-" * 90)
    out = []
    for z in [0.510, 0.706]:
        DML, DHL = predict(L['H0'], L['Om'], 0, 0, 0, z, rd_L)
        DMP, DHP = predict(P['H0'], P['Om'], P['A'], P['zc'], P['w'], z, rd_P)
        dDM = 100*(DMP-DML)/DML
        dDH = 100*(DHP-DHL)/DHL
        print(f"{z:<8.3f}{DML:<14.4f}{DMP:<16.4f}{dDM:<+8.3f}"
              f"{DHL:<14.4f}{DHP:<16.4f}{dDH:<+8.3f}")
        out.append(dict(z=z, LCDM_DM_rd=DML, LockPhi_DM_rd=DMP,
                        LCDM_DH_rd=DHL, LockPhi_DH_rd=DHP,
                        delta_DM_pct=dDM, delta_DH_pct=dDH))
    save_json(out, 'v16_DR3pred.json')
    return out


# ============================================================================
# UTIL
# ============================================================================
def save_json(obj, name):
    path = os.path.join(OUT_DIR, name)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"  saved: {path}")


def load_baseline_if_exists():
    path = os.path.join(OUT_DIR, 'v16_baseline.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="LVC v16 reproduction script")
    parser.add_argument('--baseline', action='store_true',
                        help='§2-§5.1 baseline fits (~10 min)')
    parser.add_argument('--ladder', action='store_true',
                        help='§5.2 z_c natural-constant ladder (~25 min)')
    parser.add_argument('--landscape', action='store_true',
                        help='§5.3 2D (z_c, w) landscape (~10 min)')
    parser.add_argument('--ka', action='store_true',
                        help='§6.1 K-A multi-LOO on combined BAO (~25 min)')
    parser.add_argument('--kills', action='store_true',
                        help='§6.2-§6.4 free-OL + fσ8 + BBN (~15 min)')
    parser.add_argument('--decomp', action='store_true',
                        help='§6.5 χ² decomp + A-profile (~5 min)')
    parser.add_argument('--predict', action='store_true',
                        help='§7 DR3 predictions (no fits, <1 min)')
    args = parser.parse_args()

    # If no flags, run everything
    run_all = not any([args.baseline, args.ladder, args.landscape, args.ka,
                       args.kills, args.decomp, args.predict])

    print("=" * 78)
    print("LVC v16 — single-file reproduction")
    print("=" * 78)
    print(f"  N_combined = {N_COMBINED}")
    print(f"  Output dir = {OUT_DIR}")
    print(f"  PP dir     = {PP_DIR}")
    print()

    pp = load_pantheonplus()
    print(f"  Pantheon+ loaded: N = {pp['N']}")
    baseline = load_baseline_if_exists()

    # Run sections
    if run_all or args.baseline:
        baseline = baseline_fits(pp)

    if run_all or args.ladder:
        zc_ladder(pp)

    if run_all or args.landscape:
        chi_LCDM = baseline['LCDM']['chi2'] if baseline else None
        landscape_2d(pp, chi_LCDM=chi_LCDM)

    if run_all or args.ka:
        ka_combined(pp)

    if run_all or args.kills:
        if baseline is None:
            print("\n(Running baseline first since not cached ...)")
            baseline = baseline_fits(pp)
        free_OL_test(pp, baseline=baseline)
        fsigma8_joint(pp, baseline=baseline)
        bbn_test(pp, baseline=baseline)

    if run_all or args.decomp:
        if baseline is None:
            baseline = baseline_fits(pp)
        chi2_decomp(pp, baseline=baseline)
        Aprofile_LRG12(pp)

    if run_all or args.predict:
        if baseline is None:
            baseline = baseline_fits(pp)
        dr3_predictions(baseline)

    print("\nDone.")


if __name__ == '__main__':
    main()
