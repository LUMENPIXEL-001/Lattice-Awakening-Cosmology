"""
LVC v15 Appendix — Single-file reproduction script
====================================================

Reproduces all numerical results in:
  "Appendix to v15: Multi-LOO Stress and fσ8 Joint Analysis of LockB and LockC"

This is a single self-contained Python file. The only external dependency is
the Pantheon+ data and covariance files (33 MB total) which must be in
PP_data/ relative to this script, or pointed to by PANTHEONPLUS_DIR
environment variable.

If the PP data is not present, this script will offer to download it
automatically from github.com/PantheonPlusSH0ES/DataRelease (requires git
and ~33 MB disk space).

What this script reproduces:
  1. v14 environment validation (Table 1 of appendix, χ² match to 0.005)
     - LCDM, C1 on DR2 (v14 Task 2)
     - LCDM, C1 on DR1 (v15 §2 burn at ΔBIC = +0.61)
     - LockB, LockC on DR2 + DR1 (v15 §5)
  2. K-A multi-point leave-one-out sweep (Tables 2a, 2b)
     - 11 BAO subset combinations × 2 BAO releases = 22 fits
     - LockC at zc = 4/7, w = 2/(5π)
  3. fσ8 joint analysis (Tables 3, 4)
     - 19-point fσ8 compilation from v11 §3
     - linear growth ODE on locked H(z)
     - sigma8(0) profiled, joint ΔBIC computed

Approximate runtime: 60-90 minutes on a single CPU core.

Usage:
    python lvc_v15_appendix_reproduce.py            # full reproduction
    python lvc_v15_appendix_reproduce.py --quick    # only environment validation
    python lvc_v15_appendix_reproduce.py --sniper   # only K-A sweep
    python lvc_v15_appendix_reproduce.py --fsigma8  # only fσ8 analysis

External data:
    Pantheon+SH0ES.dat
    Pantheon+SH0ES_STAT+SYS.cov
  from github.com/PantheonPlusSH0ES/DataRelease
  subdirectory: Pantheon+_Data/4_DISTANCES_AND_COVAR

Set PANTHEONPLUS_DIR environment variable, or place data in ./PP_data/

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
# SECTION 1: CONSTANTS
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

# Lock definitions
PI = np.pi
ZC_C1    = 1.0/np.sqrt(3) + 1.0/(3*PI)   # ≈ 0.6835  (v12 family n=1)
W_C1     = 2.0/(3*PI)                    # ≈ 0.2122
ZC_LOCKB = 1.0/np.sqrt(PI)               # ≈ 0.5642
W_LOCKB  = 2.0/(5*PI)                    # ≈ 0.1273
ZC_LOCKC = 4.0/7.0                       # ≈ 0.5714
W_LOCKC  = 2.0/(5*PI)                    # = W_LOCKB

# Planck 2018 compressed distance priors (Chen, Huang & Wang 2019, Table 1,
#   "TT,TE,EE+lowE+lensing")
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


# ============================================================================
# SECTION 2: PP data loader (with auto-download)
# ============================================================================

PP_DIR = os.environ.get(
    "PANTHEONPLUS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "PP_data"))
PP_DAT = os.path.join(PP_DIR, "Pantheon+SH0ES.dat")
PP_COV = os.path.join(PP_DIR, "Pantheon+SH0ES_STAT+SYS.cov")


def auto_download_pp():
    """Sparse-checkout PP data from github.com/PantheonPlusSH0ES/DataRelease.

    Requires git. Downloads ~33 MB to PP_DIR.
    """
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
    # Read covariance file: first line = N, then N*N entries
    with open(PP_COV) as f:
        N_cov = int(f.readline().strip())
        if N_cov != out['N']:
            raise RuntimeError(f"PP N mismatch: data {out['N']}, cov {N_cov}")
        cov_flat = np.fromfile(f, sep=' ', count=N_cov*N_cov)
    cov = cov_flat.reshape(N_cov, N_cov)
    out['Cinv'] = np.linalg.inv(cov)
    return out


# ============================================================================
# SECTION 3: Background and growth
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
# SECTION 4: Pantheon+ likelihood
# ============================================================================

def chi2_panplus(H0, Om, T_func, pp):
    """Pantheon+ unbinned chi^2 with M_B profiled analytically."""
    DC = comoving_dist_grid(H0, Om, T_func)
    if DC is None:
        return 1e10, 0.0
    # Distance modulus prediction
    DM_at_zHD = np.interp(pp['zHD'], _Z_GRID, DC)
    mu_pred = np.where(
        pp['is_calib'],
        pp['ceph'],
        5*np.log10((1+pp['zHEL']) * DM_at_zHD) + 25)
    d = pp['mB'] - mu_pred  # missing M_B
    
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
# SECTION 5: BAO data and likelihood (DR1 + DR2)
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


def chi2_bao(H0, Om, T_func, rd, bao_set='DR2', skip_labels=None):
    """BAO chi^2 for DR2 or DR1 BAO + always BOSS + eBOSS.
    
    skip_labels: set of labels to remove from likelihood (for LOO).
    """
    skip = set(skip_labels) if skip_labels else set()
    
    DC = comoving_dist_grid(H0, Om, T_func)
    if DC is None: return 1e10
    if T_func is None:
        H = H0 * E_lcdm(_Z_GRID, Om)
    else:
        H = H0 * E_lcdm(_Z_GRID, Om) * T_func(_Z_GRID)
    chi2 = 0.0
    
    # DESI BGS (DV)
    if bao_set == 'DR2':
        DV_data = DESI_DR2_DV
        PAIR_data = DESI_DR2_PAIR
    elif bao_set == 'DR1':
        DV_data = DESI_DR1_DV
        PAIR_data = DESI_DR1_PAIR
    else:
        raise ValueError(f"unknown bao_set {bao_set}")
    
    if 'BGS' not in skip:
        for z, DVo, sig in DV_data:
            DM = np.interp(z, _Z_GRID, DC)
            Hv = np.interp(z, _Z_GRID, H)
            DV = (z * DM**2 * C_KMS / Hv) ** (1/3)
            chi2 += ((DVo - DV/rd) / sig)**2
    
    # DESI anisotropic pairs
    for label, z, DMo, sDM, DHo, sDH, corr in PAIR_data:
        if label in skip: continue
        DMp = np.interp(z, _Z_GRID, DC) / rd
        DHp = C_KMS / np.interp(z, _Z_GRID, H) / rd
        cov = np.array([[sDM**2, corr*sDM*sDH],
                        [corr*sDM*sDH, sDH**2]])
        d = np.array([DMo - DMp, DHo - DHp])
        chi2 += d @ np.linalg.inv(cov) @ d
    
    # BOSS DR12
    if 'BOSS' not in skip:
        for (z, DMo, sDM), (z2, DHo, sDH, corr) in zip(BOSS_DM, BOSS_DH):
            DMp = np.interp(z, _Z_GRID, DC) / rd
            DHp = C_KMS / np.interp(z, _Z_GRID, H) / rd
            cov = np.array([[sDM**2, corr*sDM*sDH],
                            [corr*sDM*sDH, sDH**2]])
            d = np.array([DMo - DMp, DHo - DHp])
            chi2 += d @ np.linalg.inv(cov) @ d
    
    # eBOSS pairs
    for label, z, DMo, sDM, DHo, sDH, corr in EBOSS_DM_PAIRS:
        if label in skip: continue
        DMp = np.interp(z, _Z_GRID, DC) / rd
        DHp = C_KMS / np.interp(z, _Z_GRID, H) / rd
        cov = np.array([[sDM**2, corr*sDM*sDH],
                        [corr*sDM*sDH, sDH**2]])
        d = np.array([DMo - DMp, DHo - DHp])
        chi2 += d @ np.linalg.inv(cov) @ d
    
    # eBOSS ELG (DV)
    for label, z, DVo, sig in EBOSS_DV:
        if label in skip: continue
        DM = np.interp(z, _Z_GRID, DC)
        Hv = np.interp(z, _Z_GRID, H)
        DV = (z * DM**2 * C_KMS / Hv) ** (1/3)
        chi2 += ((DVo - DV/rd) / sig)**2
    
    return float(chi2)


# ============================================================================
# SECTION 6: Distance priors (R, l_A, omega_b)
# ============================================================================

# Calibration: at LCDM Planck fiducial (H0=67.36, Om=0.3153, omega_b=0.02237)
# we should recover R_PLANCK, LA_PLANCK exactly. Compute the offset and
# define multiplicative correction factors.
def _z_star_HS(Ob_h2, Om_h2):
    """Hu-Sugiyama (1996) approximation for z_star."""
    g1 = 0.0783 * Ob_h2**(-0.238) / (1 + 39.5 * Ob_h2**0.763)
    g2 = 0.560 / (1 + 21.1 * Ob_h2**1.81)
    return 1048 * (1 + 0.00124 * Ob_h2**(-0.738)) * (1 + g1 * Om_h2**g2)


def _R_lA_raw(H0, Om, Ob_h2, T_func=None):
    """Compute uncalibrated (R, l_A, DM_star, rs, z_star) for the model.
    Following v14 reproduce exactly. T_func takes scalar z, returns scalar T.
    """
    h = H0/100.0
    Om_h2 = Om * h * h
    z_star = _z_star_HS(Ob_h2, Om_h2)
    Or_frac = OMEGA_R_H2 / h**2
    
    if T_func is None:
        def H_E(z):
            return H0 * np.sqrt(Om*(1+z)**3 + Or_frac*(1+z)**4
                                + (1-Om-Or_frac))
    else:
        def H_E(z):
            T_val = float(T_func(np.array([z]))[0])
            return H0 * np.sqrt(Om*(1+z)**3 + Or_frac*(1+z)**4
                                + (1-Om-Or_frac) * T_val * T_val)
    
    DM_star, _ = quad(lambda z: C_KMS/H_E(z), 0, z_star, limit=400)
    rs, _ = quad(
        lambda z: 1.0/np.sqrt(3*(1 + 3*Ob_h2/(4*OMEGA_GAMMA_H2)/(1+z)))
                  * C_KMS/H_E(z),
        z_star, 1e6, limit=400)
    R_raw  = np.sqrt(Om_h2) * DM_star * 100.0 / C_KMS
    lA_raw = PI * DM_star / rs
    return R_raw, lA_raw, DM_star, rs, z_star


# Compute calibration once (LCDM at Planck fiducial should give exactly the
# Chen+19 reported R, l_A; remove ~0.5% fitting-formula offset)
_R_AT_PLANCK, _LA_AT_PLANCK, _, _, _ = _R_lA_raw(67.36, 0.3153, 0.02237, None)
CAL_R = R_PLANCK / _R_AT_PLANCK
CAL_LA = LA_PLANCK / _LA_AT_PLANCK


def chi2_DP(H0, Om, Ob_h2, T_func=None):
    """Calibrated distance-prior chi^2 against (R_PLANCK, LA_PLANCK, WB_PLANCK)."""
    R_raw, lA_raw, _, _, _ = _R_lA_raw(H0, Om, Ob_h2, T_func)
    R_cal = R_raw * CAL_R
    lA_cal = lA_raw * CAL_LA
    x = np.array([R_cal, lA_cal, Ob_h2])
    xref = np.array([R_PLANCK, LA_PLANCK, WB_PLANCK])
    d = x - xref
    return float(d @ COVINV_DP @ d), R_cal, lA_cal


def DM_at_zstar(H0, Om, Ob_h2, T_func=None):
    """Compute D_M(z_star) for r_d derivation via theta_*."""
    _, _, DM_star, _, _ = _R_lA_raw(H0, Om, Ob_h2, T_func)
    return DM_star


# ============================================================================
# SECTION 7: Total chi^2 and fitter
# ============================================================================

def chi2_total(H0, Om, Ob_h2, T_func, pp, bao_set='DR2',
               include_DP=True, skip_labels=None, return_parts=False):
    """Full likelihood: PP + BAO + SH0ES + DP (optional)."""
    DM_star = DM_at_zstar(H0, Om, Ob_h2, T_func)
    rd = THETA_STAR * DM_star
    
    chi_pp, _ = chi2_panplus(H0, Om, T_func, pp)
    chi_bao = chi2_bao(H0, Om, T_func, rd, bao_set=bao_set,
                       skip_labels=skip_labels)
    chi_h = ((H0_SHOES - H0) / SIG_H0_SHOES)**2
    chi_dp = 0.0
    R_cal, lA_cal = 0, 0
    if include_DP:
        chi_dp, R_cal, lA_cal = chi2_DP(H0, Om, Ob_h2, T_func)
    
    total = chi_pp + chi_bao + chi_h + chi_dp
    if return_parts:
        return total, dict(pp=chi_pp, bao=chi_bao, h=chi_h, dp=chi_dp,
                           rd=rd, R=R_cal, lA=lA_cal)
    return total


def _fit(cost_fn, bounds, seeds=(42, 7, 13)):
    """Multi-seed differential evolution + Nelder-Mead polish."""
    best = (1e10, None)
    for sd in seeds:
        de = differential_evolution(cost_fn, bounds, seed=sd, tol=1e-8,
                                    maxiter=200, polish=False, workers=1)
        nm = minimize(cost_fn, de.x, method='Nelder-Mead',
                      options=dict(xatol=1e-8, fatol=1e-8, maxiter=5000))
        if nm.fun < best[0]:
            best = (nm.fun, nm.x.copy())
    return best


def fit_LCDM(pp, bao_set='DR2', include_DP=True):
    """Fit LCDM (H0, Om, [Ob_h2 if DP]). Returns (chi2, params)."""
    if include_DP:
        bounds = [(60, 80), (0.20, 0.45), (0.020, 0.025)]
        def cost(p):
            H0, Om, Ob = p
            return chi2_total(H0, Om, Ob, None, pp, bao_set=bao_set,
                              include_DP=True)
    else:
        bounds = [(60, 80), (0.20, 0.45)]
        def cost(p):
            H0, Om = p
            return chi2_total(H0, Om, WB_PLANCK, None, pp, bao_set=bao_set,
                              include_DP=False)
    return _fit(cost, bounds)


def fit_locked(pp, zc, w, bao_set='DR2', include_DP=True, skip_labels=None):
    """Fit (H0, Om, Ob_h2, A) at fixed (zc, w)."""
    if include_DP:
        bounds = [(60, 80), (0.20, 0.45), (0.020, 0.025), (-0.20, 0.20)]
        def cost(p):
            H0, Om, Ob, A = p
            T = lambda z: T_gauss(z, A, zc, w)
            return chi2_total(H0, Om, Ob, T, pp, bao_set=bao_set,
                              include_DP=True, skip_labels=skip_labels)
    else:
        bounds = [(60, 80), (0.20, 0.45), (-0.20, 0.20)]
        def cost(p):
            H0, Om, A = p
            T = lambda z: T_gauss(z, A, zc, w)
            return chi2_total(H0, Om, WB_PLANCK, T, pp, bao_set=bao_set,
                              include_DP=False, skip_labels=skip_labels)
    return _fit(cost, bounds)


# ============================================================================
# SECTION 8: Linear growth ODE for fσ8
# ============================================================================

# 19-point fσ8 compilation (v11 §3)
FSIGMA8_DATA = [
    (0.067, 0.423, 0.055),  (0.150, 0.490, 0.145),  (0.180, 0.360, 0.090),
    (0.380, 0.440, 0.060),  (0.32,  0.473, 0.041),  (0.57,  0.467, 0.045),
    (0.44,  0.413, 0.080),  (0.60,  0.390, 0.063),  (0.73,  0.437, 0.072),
    (0.60,  0.55,  0.12 ),  (0.86,  0.40,  0.11 ),  (0.698, 0.473, 0.044),
    (0.85,  0.315, 0.095),  (1.48,  0.462, 0.045),  (2.334, 0.402, 0.099),
    (0.510, 0.450, 0.040),  (0.706, 0.470, 0.045),  (0.930, 0.435, 0.045),
    (1.317, 0.388, 0.055),
]
FS8_Z   = np.array([d[0] for d in FSIGMA8_DATA])
FS8_OBS = np.array([d[1] for d in FSIGMA8_DATA])
FS8_ERR = np.array([d[2] for d in FSIGMA8_DATA])


def E_full(z, Om, T_func=None):
    """E(z) including (optionally) T(z)."""
    if T_func is None:
        return E_lcdm(z, Om)
    return E_lcdm(z, Om) * T_func(z)


def growth_ODE(z_eval, Om, T_func=None):
    """Solve scale-independent linear-growth ODE.

    d²D/d(lna)² + (2 + dlnE/dlna) dD/dlna - (3/2) Ωm(a) D = 0
    IC: D = a, dD/dlna = a at a_init = 1e-3
    Returns (f(z), D(z)/D(0)).
    """
    a_init = 1e-3
    def Om_at_a(a):
        z = 1.0/a - 1.0
        E = E_full(z, Om, T_func)
        return Om*(1+z)**3 / E**2
    def dlnE_dlna(a, eps=1e-5):
        ap, am = a*np.exp(eps), a*np.exp(-eps)
        zp, zm = 1/ap - 1, 1/am - 1
        Ep = E_full(zp, Om, T_func)
        Em = E_full(zm, Om, T_func)
        return (np.log(Ep) - np.log(Em)) / (2*eps)
    def rhs(lna, y):
        D, dD = y
        a = np.exp(lna)
        Oma  = Om_at_a(a)
        dlnE = dlnE_dlna(a)
        ddD  = -(2+dlnE)*dD + 1.5*Oma*D
        return [dD, ddD]
    
    z_extended = sorted(set(list(z_eval) + [0.0]))
    a_q = sorted([1.0/(1+zz) for zz in z_extended if 1.0/(1+zz) > a_init])
    lna_q = [np.log(aq) for aq in a_q]
    
    sol = solve_ivp(rhs, [np.log(a_init), 0.0], [a_init, a_init],
                    t_eval=lna_q, method='RK45',
                    rtol=1e-8, atol=1e-10, max_step=0.05)
    if not sol.success:
        return None, None
    
    D_grid = sol.y[0]
    f_grid = sol.y[1] / sol.y[0]
    idx0 = np.argmin(np.abs(np.exp(sol.t) - 1.0))
    D0 = D_grid[idx0]
    
    a_eval = np.array([1.0/(1+zz) for zz in z_eval])
    f_e = np.interp(np.log(a_eval), sol.t, f_grid)
    D_e = np.interp(np.log(a_eval), sol.t, D_grid) / D0
    return f_e, D_e


def chi2_fsigma8(Om, sigma8_0, T_func=None):
    f_e, D_e = growth_ODE(FS8_Z, Om, T_func)
    if f_e is None:
        return 1e10
    pred = f_e * sigma8_0 * D_e
    return float(np.sum(((FS8_OBS - pred) / FS8_ERR)**2))


def fit_sigma8(Om, T_func=None):
    """Profile sigma8(0) at fixed background."""
    res = minimize_scalar(lambda s8: chi2_fsigma8(Om, s8, T_func),
                          bounds=(0.5, 1.2), method='bounded',
                          options=dict(xatol=1e-5))
    return res.x, res.fun


# ============================================================================
# SECTION 9: K-A multi-LOO sniper
# ============================================================================

KA_COMBOS = [
    ('LRG1',),
    ('LRG2',),
    ('eBOSS_LRG',),
    ('BOSS',),
    ('LRG1', 'LRG2'),
    ('LRG1', 'eBOSS_LRG'),
    ('LRG1', 'BOSS'),
    ('LRG1', 'LRG2', 'eBOSS_LRG'),
    ('LRG1', 'LRG2', 'BOSS'),
    ('LRG1', 'BOSS', 'eBOSS_LRG'),
    ('LRG1', 'LRG2', 'BOSS', 'eBOSS_LRG'),
]

LABEL_NMEAS = {
    'BGS': 1, 'LRG1': 2, 'LRG2': 2, 'LRG3': 2, 'ELG2': 2, 'QSO': 2, 'Lya': 2,
    'BOSS': 2, 'eBOSS_LRG': 2, 'eBOSS_QSO': 2, 'eBOSS_Lya': 2, 'eBOSS_ELG': 1,
}

N_DR2_FULL = 1727
N_DR1_FULL = 1725


def run_ka_sweep(pp, zc, w, bao_set='DR2', verbose=True):
    """K-A: 11 multi-LOO subsets at given lock. Returns dict."""
    results = {}
    N_full = N_DR2_FULL if bao_set == 'DR2' else N_DR1_FULL
    
    # First: LCDM and lock at full data
    if verbose:
        print(f"  [{bao_set}] baseline LCDM ...", end='', flush=True)
    chi_L, p_L = fit_LCDM(pp, bao_set=bao_set, include_DP=True)
    if verbose:
        print(f" χ²={chi_L:.3f}")
    
    for combo in KA_COMBOS:
        skip_set = set(combo)
        dN = sum(LABEL_NMEAS[lab] for lab in combo)
        N_eff = N_full - dN
        
        if verbose:
            print(f"  [{bao_set}] removing {'+'.join(combo)} (dN={dN}) ...",
                  end='', flush=True)
        t0 = time.time()
        
        # Refit LCDM and lock with these labels removed
        # LCDM
        bounds_L = [(60, 80), (0.20, 0.45), (0.020, 0.025)]
        def cost_L(p):
            H0, Om, Ob = p
            return chi2_total(H0, Om, Ob, None, pp, bao_set=bao_set,
                              include_DP=True, skip_labels=skip_set)
        chi_L_loo, _ = _fit(cost_L, bounds_L, seeds=(42, 7))
        
        # Lock
        bounds_lk = [(60, 80), (0.20, 0.45), (0.020, 0.025), (-0.20, 0.20)]
        def cost_lk(p):
            H0, Om, Ob, A = p
            T = lambda z: T_gauss(z, A, zc, w)
            return chi2_total(H0, Om, Ob, T, pp, bao_set=bao_set,
                              include_DP=True, skip_labels=skip_set)
        chi_lk, p_lk = _fit(cost_lk, bounds_lk, seeds=(42, 7))
        H0, Om, Ob, A = p_lk
        
        dchi = chi_lk - chi_L_loo
        dBIC = dchi + np.log(N_eff)
        verdict = ('WIN' if dBIC < -2 else 'tie' if dBIC < 2 else 'BURN')
        
        results['+'.join(combo)] = dict(
            combo=list(combo), dN=dN, N_eff=N_eff,
            chi2_L=chi_L_loo, chi2_lock=chi_lk, A=A,
            dchi=dchi, dBIC=dBIC, verdict=verdict, sec=time.time()-t0)
        if verbose:
            print(f" Δχ²={dchi:+.3f} ΔBIC={dBIC:+.3f} {verdict}"
                  f" [{time.time()-t0:.0f}s]")
    
    return results


# ============================================================================
# SECTION 10: Main reproduction routines
# ============================================================================

def reproduce_environment(pp, save_path):
    """§1 — Environment validation: LCDM, C1, LockB, LockC on DR2 + DR1."""
    print("\n" + "=" * 78)
    print("§1 ENVIRONMENT VALIDATION (Table 1)")
    print("=" * 78)
    
    results = {}
    targets = {
        ('LCDM','DR2'): None,
        ('LCDM','DR1'): None,
        ('C1','DR2'):    -11.38,
        ('C1','DR1'):     0.61,
        ('LockB','DR2'): -5.29,
        ('LockB','DR1'): -3.60,
        ('LockC','DR2'): -6.21,
        ('LockC','DR1'): -3.60,
    }
    LOCKS = {
        'C1':    (ZC_C1,    W_C1),
        'LockB': (ZC_LOCKB, W_LOCKB),
        'LockC': (ZC_LOCKC, W_LOCKC),
    }
    
    # LCDM first
    for bao in ('DR2', 'DR1'):
        print(f"\n  LCDM on {bao} ...", flush=True)
        t0 = time.time()
        chi, p = fit_LCDM(pp, bao_set=bao, include_DP=True)
        H0, Om, Ob = p
        results.setdefault('LCDM', {})[bao] = dict(
            chi2=float(chi), H0=float(H0), Om=float(Om), Ob=float(Ob))
        print(f"    χ²={chi:.3f}  H0={H0:.3f}  Om={Om:.4f}  Ob={Ob:.5f}  "
              f"[{time.time()-t0:.0f}s]")
    
    # Locked candidates
    for name, (zc, w) in LOCKS.items():
        for bao in ('DR2', 'DR1'):
            print(f"\n  {name} on {bao} (zc={zc:.4f}, w={w:.4f}) ...",
                  flush=True)
            t0 = time.time()
            chi, p = fit_locked(pp, zc, w, bao_set=bao, include_DP=True)
            H0, Om, Ob, A = p
            chi_L = results['LCDM'][bao]['chi2']
            N = N_DR2_FULL if bao == 'DR2' else N_DR1_FULL
            dchi = chi - chi_L
            dBIC = dchi + np.log(N)
            target = targets.get((name, bao))
            target_str = f"  [target {target:+.2f}]" if target is not None else ""
            print(f"    χ²={chi:.3f}  H0={H0:.3f}  Om={Om:.4f}  "
                  f"A={A:+.5f}  ΔBIC={dBIC:+.3f}{target_str}  "
                  f"[{time.time()-t0:.0f}s]")
            results.setdefault(name, {})[bao] = dict(
                chi2=float(chi), H0=float(H0), Om=float(Om), Ob=float(Ob),
                A=float(A), dchi=float(dchi), dBIC=float(dBIC))
    
    # Save and print summary
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {save_path}")
    
    # Sanity check vs targets
    print("\n  v15 paper match check:")
    print(f"    {'Model':<8} {'BAO':<5} {'ours':<10} {'target':<10} {'diff':<10}")
    for (m, b), tgt in targets.items():
        if tgt is None: continue
        ours = results[m][b]['dBIC']
        diff = ours - tgt
        ok = "✓" if abs(diff) < 0.05 else "✗"
        print(f"    {m:<8} {b:<5} {ours:+10.3f} {tgt:+10.2f} {diff:+8.4f} {ok}")
    
    return results


def reproduce_ka_sweep(pp, save_path):
    """§2 — K-A multi-LOO for LockC on DR2 + DR1 (Tables 2a, 2b)."""
    print("\n" + "=" * 78)
    print("§2 K-A MULTI-LOO SWEEP (Tables 2a, 2b)")
    print("=" * 78)
    
    results = {'LockC': {}}
    for bao in ('DR2', 'DR1'):
        print(f"\n  LockC on {bao} — 11 subsets")
        results['LockC'][bao] = run_ka_sweep(pp, ZC_LOCKC, W_LOCKC,
                                              bao_set=bao, verbose=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved progress: {save_path}")
    
    # Print summary
    print("\n  Sorted by ΔBIC:")
    for bao in ('DR2', 'DR1'):
        print(f"\n  --- LockC {bao} ---")
        items = sorted(results['LockC'][bao].items(),
                       key=lambda x: x[1]['dBIC'])
        for name, d in items:
            print(f"    {name:<35} dN={d['dN']}  ΔBIC={d['dBIC']:+8.3f}  "
                  f"{d['verdict']}")
    return results


def reproduce_fsigma8(pp, env_results, save_path):
    """§3 — fσ8 joint analysis (Tables 3, 4)."""
    print("\n" + "=" * 78)
    print("§3 fσ8 JOINT ANALYSIS (Tables 3, 4)")
    print("=" * 78)
    
    # Sanity: LCDM at Planck values
    chi_san = chi2_fsigma8(0.315, 0.811, None)
    print(f"\n  Sanity: LCDM(Om=0.315, σ8=0.811): χ²(fσ8)={chi_san:.3f} "
          f"(v11 expects 11.99)")
    
    results = {}
    LOCKS = {
        'LCDM':  None,
        'LockB': (ZC_LOCKB, W_LOCKB),
        'LockC': (ZC_LOCKC, W_LOCKC),
    }
    
    print(f"\n  --- Table 3: fσ8-only fits ---")
    print(f"  {'Model':<7} {'BAO':<4} {'Om':<8} {'σ8':<8} "
          f"{'χ²(fσ8)':<10} {'Δχ² vs LCDM'}")
    
    for name, lock_params in LOCKS.items():
        for bao in ('DR2', 'DR1'):
            Om = env_results[name][bao]['Om']
            if lock_params is None:
                T = None
            else:
                A = env_results[name][bao]['A']
                zc, w = lock_params
                T = lambda z, A=A, zc=zc, w=w: T_gauss(z, A, zc, w)
            s8, chi_fs8 = fit_sigma8(Om, T)
            chi_L_fs8 = (results.get('LCDM', {}).get(bao, {})
                         .get('chi2_fs8', None))
            d = chi_fs8 - chi_L_fs8 if chi_L_fs8 is not None else None
            d_str = f"{d:+.3f}" if d is not None else "—"
            print(f"  {name:<7} {bao:<4} {Om:<8.4f} {s8:<8.4f} "
                  f"{chi_fs8:<10.3f} {d_str}")
            results.setdefault(name, {})[bao] = dict(
                Om=Om, sigma8=s8, chi2_fs8=chi_fs8,
                dchi_vs_LCDM=d if d is not None else 0.0)
    
    # Joint analysis
    print(f"\n  --- Table 4: Geometry + fσ8 joint ---")
    print(f"  {'Model':<7} {'BAO':<4} {'χ²(geom)':<11} {'χ²(fσ8)':<10} "
          f"{'Total':<10} {'ΔBIC vs LCDM'}")
    
    for name in ('LCDM', 'LockB', 'LockC'):
        for bao in ('DR2', 'DR1'):
            chi_geom = env_results[name][bao]['chi2']
            chi_fs8 = results[name][bao]['chi2_fs8']
            total = chi_geom + chi_fs8
            N = (1727 if bao == 'DR2' else 1725) + 19
            results[name][bao]['chi2_total_joint'] = total
            if name == 'LCDM':
                ref = total
                print(f"  {name:<7} {bao:<4} {chi_geom:<11.3f} {chi_fs8:<10.3f} "
                      f"{total:<10.3f} (baseline)")
            else:
                ref = (env_results['LCDM'][bao]['chi2']
                       + results['LCDM'][bao]['chi2_fs8'])
                dchi = total - ref
                dBIC = dchi + np.log(N)  # +1 parameter (A)
                results[name][bao]['dBIC_joint'] = dBIC
                v = 'WIN' if dBIC < -2 else ('tie' if dBIC < 2 else 'BURN')
                print(f"  {name:<7} {bao:<4} {chi_geom:<11.3f} {chi_fs8:<10.3f} "
                      f"{total:<10.3f} {dBIC:+8.3f}  {v}")
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {save_path}")
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--quick', action='store_true',
        help='Run only environment validation (~10 min)')
    parser.add_argument('--sniper', action='store_true',
        help='Run only K-A multi-LOO sweep (~40 min, requires env results)')
    parser.add_argument('--fsigma8', action='store_true',
        help='Run only fσ8 analysis (~5 min, requires env results)')
    parser.add_argument('--outdir', default='.',
        help='Output directory for JSON results')
    args = parser.parse_args()
    
    do_all = not (args.quick or args.sniper or args.fsigma8)
    do_env    = do_all or args.quick or args.sniper or args.fsigma8
    do_sniper = do_all or args.sniper
    do_fs8    = do_all or args.fsigma8
    
    os.makedirs(args.outdir, exist_ok=True)
    env_path = os.path.join(args.outdir, 'v15app_env_results.json')
    ka_path  = os.path.join(args.outdir, 'v15app_ka_results.json')
    fs8_path = os.path.join(args.outdir, 'v15app_fs8_results.json')
    
    print("=" * 78)
    print("LVC v15 Appendix — Reproduction Script")
    print("=" * 78)
    print(f"PP_DIR: {PP_DIR}")
    print(f"Output: {args.outdir}")
    print()
    
    print("Loading Pantheon+ unbinned data + STAT+SYS covariance ...")
    t0 = time.time()
    pp = load_pantheonplus()
    print(f"  N={pp['N']} SN, {pp['is_calib'].sum()} Cepheid calibrators "
          f"[{time.time()-t0:.1f}s]")
    
    # Sanity: LCDM at (73.04, 0.30) should give χ² ≈ 1526.49
    chi_san, M_san = chi2_panplus(73.04, 0.30, None, pp)
    print(f"  Sanity: LCDM(73.04, 0.30) χ²(PP)={chi_san:.2f} "
          f"(v14 expects 1526.49)")
    
    env_results = None
    if do_env:
        if os.path.exists(env_path) and not do_all:
            print(f"\n  Loading cached environment results: {env_path}")
            with open(env_path) as f:
                env_results = json.load(f)
        else:
            env_results = reproduce_environment(pp, env_path)
    
    if do_sniper:
        reproduce_ka_sweep(pp, ka_path)
    
    if do_fs8:
        if env_results is None:
            with open(env_path) as f:
                env_results = json.load(f)
        reproduce_fsigma8(pp, env_results, fs8_path)
    
    print("\n" + "=" * 78)
    print(f"Done. Total runtime: {(time.time()-t0)/60:.1f} min")
    print("=" * 78)


if __name__ == '__main__':
    main()
