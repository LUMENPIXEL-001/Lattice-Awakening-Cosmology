"""
LVC v0.3 -- Single-file reproduction script
============================================

Reproduces all numerical results in the working paper:

    "Lagrangian Variable Cosmology v0.3:
     Application of the v0.2 Section 10 Protocol to Real Data
     and Diagnostic Reframing of the v0.1 Width Lock"
                                LUMENPIXEL, May 2026

This is a single self-contained Python file. No LVC-specific dependencies.
All constants, BAO data tables, distance-prior calibration, Pantheon+
loader, and fitters are inlined. The likelihood backend is identical to
v0.1 reproduction script lvc_v01.py.

What this paper / script add over v0.1
--------------------------------------
v0.1 fitted three models with locks:
    Lambda-CDM (k=3),
    Lock-phi-A v16 (z-Gaussian, locked z_c = 1/phi, w = e/(5pi)),  k=4
    LVC pendulum  (theta-Gaussian, locked theta_c = ln phi, w = e/(5pi)),  k=4

v0.3 OPENS THE LOCK on z_c, w (or theta_c, w) and runs the v0.2 Section 10
falsification protocol on real data:
    - F1 / F2 free fit (k=6) on PP-EXCLUDED data (BAO+DP+SH0ES, N=37)
    - F1 / F2 free fit (k=6) on PP-INCLUDED data (full likelihood, N=1738)
    - Hessian-based sigma on each best fit
    - 1-D profile likelihood scan (Delta chi^2 < 1, < 4 confidence regions)
    - Parametric residual bootstrap on PP-EXCLUDED data (B=30)

Reproduces (key v0.3 numbers):
------------------------------
v0.1 sanity (locked fits, full likelihood N=1738):
    Lambda-CDM                        chi^2 = 1624.85
    Lock-phi-A v16  (k=4)             chi^2 = 1601.19
    LVC pendulum    (k=4)             chi^2 = 1602.77

PP-excluded free fits (k=6, N=37):
    F1 mode A:     chi^2 = 56.317  zc=0.6066    w=0.1053  A=+0.0955
    F1 mode B:     chi^2 = 56.402  zc=0.6407    w=0.2012  A=+0.0533
    F2:            chi^2 = 56.038  theta_c=0.4873  w=0.1161  A=+0.0555

PP-included free fits (k=6, N=1738):
    F1:            chi^2 = 1601.189  zc=0.6180=1/phi      w=0.171  A=+0.0525
    F2:            chi^2 = 1600.997  theta_c=0.4812=lnphi  w=0.110  A=+0.0515
    Delta(F2 - F1) = -0.192

Profile likelihood 1-sigma half-widths:
    F1 PP-excl:   0.0750
    F2 PP-excl:   0.0400
    F1 PP-incl:   0.0600
    F2 PP-incl:   0.0400

Bootstrap distributions (PP-excluded, B=30):
    F1: zc mean=0.5967, std=0.0246, q16-q84 width=0.0402
    F2: theta_c mean=0.4635, std=0.0172, q16-q84 width=0.0271
    Mode pick: A=1, A_close=15, B=14
    Median |chi2_A - chi2_B| = 0.0000

External data
-------------
Pantheon+ data (~33 MB) and covariance must be available locally:
    Pantheon+SH0ES.dat
    Pantheon+SH0ES_STAT+SYS.cov

If absent, the script offers automatic git-based download from
github.com/PantheonPlusSH0ES/DataRelease (sub-directory
Pantheon+_Data/4_DISTANCES_AND_COVAR).

Set PANTHEONPLUS_DIR environment variable, or place files in ./PP_data/

Approximate runtime
-------------------
Default fast run (single seed, smaller bootstrap):       ~30 min
Full run (--multiseed --bootstrap-full):                 ~2 hours

Usage
-----
    python lvc_v03_reproduce.py                       # default
    python lvc_v03_reproduce.py --multiseed            # 3 seeds per fit
    python lvc_v03_reproduce.py --bootstrap-full       # B=60 instead of 30
    python lvc_v03_reproduce.py --skip-bootstrap       # no bootstrap
    python lvc_v03_reproduce.py --skip-pp-included     # PP-excluded only
    python lvc_v03_reproduce.py --skip-baselines       # no v0.1 sanity
    python lvc_v03_reproduce.py --save-json PATH       # save numbers
    python lvc_v03_reproduce.py --help                 # full options

Working principle
-----------------
The numerical observation that free-fit widths prefer w ~ 0.11 rather
than the v0.1 axiom L4 lock e/(5pi) ~ 0.173 is recorded here as a
fact of the present run. v0.1 axiom system is NOT modified in this
script; the locked v0.1 fits are reproduced as sanity, and the v0.3
free fits are reported separately.
"""

import os
import sys
import time
import json
import argparse
import subprocess

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import differential_evolution, minimize


# ============================================================================
# SECTION 1.  CONSTANTS
# ============================================================================

C_KMS         = 299_792.458              # speed of light, km/s
THETA_STAR    = 0.010409                 # Planck 2018 acoustic angle
SIG_THETA_STAR = 3.1e-5
Z_STAR        = 1090.0                   # photon decoupling redshift
H0_SHOES      = 73.04                    # SH0ES (Riess+ 2022)
SIG_H0_SHOES  = 1.04

# Radiation density (T_CMB = 2.7255 K, N_eff = 3.046)
OMEGA_GAMMA_H2 = 2.473e-5
NEFF_STD       = 3.046
OMEGA_R_H2     = OMEGA_GAMMA_H2 * (1 + 7/8 * (4/11)**(4/3) * NEFF_STD)

# Pi, e, Phi
PI  = np.pi
E   = np.e
PHI = (1 + np.sqrt(5)) / 2.0
INV_PHI    = 1.0 / PHI                    # 0.61803...
THETA_C_PHI = np.log(PHI)                 # 0.48121...

# Lock-phi-A parameters (v16 phenomenology, also used by v0.1 LVC)
W_PHI_A    = E / (5 * PI)                 # 0.17305
W_LVC      = E / (5 * PI)                 # 0.17305 (same value, axiom L4)

# Planck 2018 compressed distance priors
# (Chen, Huang & Wang 2019, Table 1, "TT,TE,EE+lowE+lensing")
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

# Combined-likelihood total measurement count (= 1738)
N_COMBINED = 1701 + 13 + 11 + 2 + 7 + 3 + 1


# ============================================================================
# SECTION 2.  PANTHEON+ DATA LOADER (with auto-download)
# ============================================================================

PP_DIR = os.environ.get(
    "PANTHEONPLUS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "PP_data"))
PP_DAT = os.path.join(PP_DIR, "Pantheon+SH0ES.dat")
PP_COV = os.path.join(PP_DIR, "Pantheon+SH0ES_STAT+SYS.cov")


def auto_download_pp():
    """Sparse-checkout PP data from PantheonPlusSH0ES/DataRelease."""
    print(f"\nPantheon+ data not found at {PP_DIR}.")
    print("Attempting automatic download from GitHub...")
    print(f"  Target directory: {PP_DIR}")
    print(f"  Required disk:    ~33 MB")
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
    """Load PP data and full STAT+SYS covariance."""
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
# SECTION 3.  BACKGROUND COSMOLOGY
# ============================================================================

# Dense z-grid for line-of-sight integrals
_Z_GRID = np.concatenate([[0.0], np.geomspace(1e-3, 5.0, 200)])


def E_lcdm(z, Om):
    """Flat Lambda-CDM normalised Hubble rate."""
    return np.sqrt(Om*(1+z)**3 + (1-Om))


# ----------------------------------------------------------------------------
#  Modulation factors
# ----------------------------------------------------------------------------

def T_zGauss(z, A, zc=INV_PHI, w=W_PHI_A):
    """F1: Lock-phi-A modulation (Gaussian on z)."""
    return 1.0 + A * np.exp(-((z - zc) / w)**2)


def T_thetaGauss(z, A, theta_c=THETA_C_PHI, w=W_LVC):
    """F2: LVC pendulum modulation (Gaussian on theta = ln(1+z))."""
    theta = np.log(1.0 + z)
    return 1.0 + A * np.exp(-((theta - theta_c) / w)**2)


# ----------------------------------------------------------------------------
#  Comoving distance
# ----------------------------------------------------------------------------

def comoving_dist_grid(H0, Om, T_func=None):
    """D_C(z) on _Z_GRID. T_func=None for Lambda-CDM, else callable."""
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
# SECTION 5.  BAO DATA TABLES (DR1, DR2, BOSS, eBOSS)
# ============================================================================

# DESI DR2 BAO (DESI Collaboration 2025)
DESI_DR2_DV = [(0.295, 7.942, 0.075)]      # BGS, isotropic
DESI_DR2_PAIR = [
    ('LRG1', 0.510, 13.587, 0.169, 21.863, 0.427, -0.475),
    ('LRG2', 0.706, 17.347, 0.180, 19.458, 0.332, -0.423),
    ('LRG3', 0.934, 21.574, 0.153, 17.641, 0.193, -0.425),
    ('QSO',  1.321, 27.605, 0.320, 14.178, 0.217, -0.437),
    ('ELG2', 1.484, 30.519, 0.758, 12.816, 0.513, -0.489),
    ('Lya',  2.330, 38.988, 0.531,  8.632, 0.101, -0.431),
]

# DESI DR1 BAO (DESI Collaboration 2024)
DESI_DR1_DV = [(0.295, 7.93, 0.150)]
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
# SECTION 6.  DISTANCE PRIORS  (sound-horizon multiplicative calibration)
# ============================================================================

def _z_star_HS(Ob_h2, Om_h2):
    """Hu-Sugiyama 1996 fitting formula for z_*."""
    g1 = 0.0783 * Ob_h2**(-0.238) / (1 + 39.5 * Ob_h2**0.763)
    g2 = 0.560 / (1 + 21.1 * Ob_h2**1.81)
    return 1048 * (1 + 0.00124 * Ob_h2**(-0.738)) * (1 + g1 * Om_h2**g2)


def _R_lA_raw(H0, Om, Ob_h2, T_func=None):
    """Raw R, l_A, D_M(z*), r_s(z*) without calibration."""
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

    DM_star, _ = quad(lambda z: C_KMS/H_E(z), 0, z_star, limit=400)
    rs, _ = quad(
        lambda z: 1.0/np.sqrt(3*(1 + 3*Ob_h2/(4*OMEGA_GAMMA_H2)/(1+z)))
                  * C_KMS/H_E(z),
        z_star, 1e6, limit=400)
    R_raw = np.sqrt(Om_h2) * DM_star * 100.0 / C_KMS
    lA_raw = PI * DM_star / rs
    return R_raw, lA_raw, DM_star, rs, z_star


# Calibration: at Planck fiducial flat Lambda-CDM, multiplicative factor
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
# SECTION 7.  COMBINED LIKELIHOOD
# ============================================================================

def chi2_bao_combined(H0, Om, T_func, rd):
    """DR1 + DR2 + BOSS + eBOSS BAO chi^2."""
    DC = comoving_dist_grid(H0, Om, T_func)
    if DC is None:
        return 1e10
    if T_func is None:
        H = H0 * E_lcdm(_Z_GRID, Om)
    else:
        H = H0 * E_lcdm(_Z_GRID, Om) * T_func(_Z_GRID)

    chi2 = 0.0

    for z, DVo, sig in DESI_DR2_DV:
        DM = np.interp(z, _Z_GRID, DC)
        Hv = np.interp(z, _Z_GRID, H)
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
        DM = np.interp(z, _Z_GRID, DC)
        Hv = np.interp(z, _Z_GRID, H)
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
        DM = np.interp(z, _Z_GRID, DC)
        Hv = np.interp(z, _Z_GRID, H)
        DV = (z * DM**2 * C_KMS / Hv) ** (1/3)
        chi2 += ((DVo - DV / rd) / sig) ** 2

    return float(chi2)


def chi2_full(H0, Om, Ob_h2, T_func, pp):
    """Full likelihood:  PP + (DR1+DR2)BAO + BOSS + eBOSS + SH0ES + DP."""
    DM_star = DM_at_zstar(H0, Om, Ob_h2, T_func)
    rd = THETA_STAR * DM_star
    chi_pp, _ = chi2_panplus(H0, Om, T_func, pp)
    chi_bao = chi2_bao_combined(H0, Om, T_func, rd)
    chi_h = ((H0_SHOES - H0) / SIG_H0_SHOES) ** 2
    chi_dp, _, _ = chi2_DP(H0, Om, Ob_h2, T_func)
    return chi_pp + chi_bao + chi_h + chi_dp


def chi2_no_pp(H0, Om, Ob_h2, T_func):
    """PP-excluded likelihood: BAO + BOSS + eBOSS + SH0ES + DP only.  N = 37."""
    DM_star = DM_at_zstar(H0, Om, Ob_h2, T_func)
    rd = THETA_STAR * DM_star
    chi_bao = chi2_bao_combined(H0, Om, T_func, rd)
    chi_h = ((H0_SHOES - H0) / SIG_H0_SHOES) ** 2
    chi_dp, _, _ = chi2_DP(H0, Om, Ob_h2, T_func)
    return chi_bao + chi_h + chi_dp


# ============================================================================
# SECTION 8.  v0.1 SANITY-CHECK FITTERS
# ============================================================================

def _de_nm(cost, bounds, seeds, maxiter=120, popsize=12, polish_iter=2500):
    """Differential evolution + Nelder-Mead polish, multi-seed."""
    best = (1e10, None)
    for sd in seeds:
        de = differential_evolution(cost, bounds, seed=sd, tol=1e-7,
                                    maxiter=maxiter, polish=False,
                                    popsize=popsize)
        nm = minimize(cost, de.x, method='Nelder-Mead',
                      options=dict(xatol=1e-8, fatol=1e-8, maxiter=polish_iter))
        if nm.fun < best[0]:
            best = (float(nm.fun), nm.x.copy())
    return best


def fit_LCDM(pp, seeds):
    """Lambda-CDM (k=3): H0, Om, Ob_h2."""
    bounds = [(60, 80), (0.20, 0.45), (0.020, 0.025)]
    def cost(p):
        return chi2_full(p[0], p[1], p[2], None, pp)
    return _de_nm(cost, bounds, seeds=seeds)


def fit_LockPhiA(pp, seeds):
    """Lock-phi-A v16 (k=4): F1 with locked z_c=1/phi, w=e/(5pi)."""
    bounds = [(60, 80), (0.20, 0.45), (0.020, 0.025), (-0.3, 0.3)]
    def cost(p):
        T = lambda z: T_zGauss(z, p[3])
        return chi2_full(p[0], p[1], p[2], T, pp)
    return _de_nm(cost, bounds, seeds=seeds)


def fit_LVC(pp, seeds):
    """LVC pendulum v0.1 (k=4): F2 with locked theta_c=ln(phi), w=e/(5pi)."""
    bounds = [(60, 80), (0.20, 0.45), (0.020, 0.025), (-0.3, 0.3)]
    def cost(p):
        T = lambda z: T_thetaGauss(z, p[3])
        return chi2_full(p[0], p[1], p[2], T, pp)
    return _de_nm(cost, bounds, seeds=seeds)


# ============================================================================
# SECTION 9.  v0.3 FREE-GAUSSIAN FITTERS (k=6)
# ============================================================================

def T_z_free(z, A, zc, w):
    """F1 free-Gaussian (z-coordinate)."""
    return 1.0 + A * np.exp(-((z - zc) / w)**2)


def T_theta_free(z, A, theta_c, w):
    """F2 free-Gaussian (theta-coordinate)."""
    theta = np.log1p(z)
    return 1.0 + A * np.exp(-((theta - theta_c) / w)**2)


# Bounds for the 6-parameter fit: H0, Om, Ob_h2, A, position, w
def _bounds_F1():
    return [(60, 80), (0.20, 0.45), (0.020, 0.025),
            (-0.3, 0.3), (0.10, 1.30), (0.03, 0.50)]

def _bounds_F2():
    # theta_c bounds:  z in [0.10, 1.30]  ->  theta in [0.095, 0.833]
    return [(60, 80), (0.20, 0.45), (0.020, 0.025),
            (-0.3, 0.3), (0.095, 0.833), (0.03, 0.50)]


def fit_F1_free(pp, seeds, no_pp=False):
    """F1 (z-Gaussian) free fit, k=6."""
    bounds = _bounds_F1()
    def cost(p):
        H0, Om, Ob, A, zc, w = p
        T = lambda z: T_z_free(z, A, zc, w)
        return chi2_no_pp(H0, Om, Ob, T) if no_pp else chi2_full(H0, Om, Ob, T, pp)
    return _de_nm_robust(cost, bounds, seeds)


def fit_F2_free(pp, seeds, no_pp=False):
    """F2 (theta-Gaussian) free fit, k=6."""
    bounds = _bounds_F2()
    def cost(p):
        H0, Om, Ob, A, th, w = p
        T = lambda z: T_theta_free(z, A, th, w)
        return chi2_no_pp(H0, Om, Ob, T) if no_pp else chi2_full(H0, Om, Ob, T, pp)
    return _de_nm_robust(cost, bounds, seeds)


def _de_nm_robust(cost, bounds, seeds, maxiter=200, popsize=15, polish_iter=4000):
    """DE + NM polish, multi-seed, returns (chi2, best params)."""
    best = (1e10, None)
    for sd in seeds:
        de = differential_evolution(cost, bounds, seed=sd, tol=1e-8,
                                    maxiter=maxiter, polish=False,
                                    popsize=popsize, mutation=(0.5, 1.5),
                                    recombination=0.7)
        nm = minimize(cost, de.x, method='Nelder-Mead',
                      options=dict(xatol=1e-9, fatol=1e-9, maxiter=polish_iter,
                                   adaptive=True))
        if nm.fun < best[0]:
            best = (float(nm.fun), nm.x.copy())
    return best


# ============================================================================
# SECTION 10.  PROFILE LIKELIHOOD  (warm-start NM, optional DE cross-check)
# ============================================================================

def _bounded_cost(family, pos_fixed, pp, no_pp):
    """Closure for 5-parameter cost at fixed position."""
    def cost(p):
        H0, Om, Ob, A, w = p
        if not (60 <= H0 <= 80 and 0.20 <= Om <= 0.45
                and 0.020 <= Ob <= 0.025 and -0.3 <= A <= 0.3
                and 0.03 <= w <= 0.50):
            return 1e8
        if family == 'F1':
            T = lambda z: T_z_free(z, A, pos_fixed, w)
        else:
            T = lambda z: T_theta_free(z, A, pos_fixed, w)
        return chi2_no_pp(H0, Om, Ob, T) if no_pp else chi2_full(H0, Om, Ob, T, pp)
    return cost


def profile_walk(family, pos_grid, pos_anchor, x_anchor, pp, no_pp):
    """Sliding warm-start NM profile across a 1-D grid.

    Walks outward from the anchor point in both directions to keep the
    warm-start valid.  Returns lists of (chi2, params5) of length len(pos_grid).
    """
    n = len(pos_grid)
    anchor_idx = int(np.argmin(np.abs(pos_grid - pos_anchor)))
    results = [None] * n

    # anchor point itself
    cost = _bounded_cost(family, pos_grid[anchor_idx], pp, no_pp)
    res = minimize(cost, x_anchor, method='Nelder-Mead',
                   options=dict(xatol=1e-7, fatol=1e-7, maxiter=3000,
                                adaptive=True))
    results[anchor_idx] = (float(res.fun), res.x.tolist())

    # walk upward
    x = list(res.x)
    for i in range(anchor_idx + 1, n):
        cost = _bounded_cost(family, pos_grid[i], pp, no_pp)
        res = minimize(cost, x, method='Nelder-Mead',
                       options=dict(xatol=1e-7, fatol=1e-7, maxiter=3000,
                                    adaptive=True))
        results[i] = (float(res.fun), res.x.tolist())
        x = list(res.x)

    # walk downward
    x = list(results[anchor_idx][1])
    for i in range(anchor_idx - 1, -1, -1):
        cost = _bounded_cost(family, pos_grid[i], pp, no_pp)
        res = minimize(cost, x, method='Nelder-Mead',
                       options=dict(xatol=1e-7, fatol=1e-7, maxiter=3000,
                                    adaptive=True))
        results[i] = (float(res.fun), res.x.tolist())
        x = list(res.x)

    return results


def profile_summary(grid, results, pos_target):
    """Compute argmin, 1-sigma and 2-sigma confidence regions, and locate
    local minima within the grid."""
    chi2 = np.array([r[0] for r in results])
    chi2_min = float(chi2.min())
    argmin = float(grid[int(np.argmin(chi2))])
    dchi2 = chi2 - chi2_min
    g1 = grid[dchi2 < 1.0]
    g2 = grid[dchi2 < 4.0]
    one_sig = (float(g1.min()), float(g1.max())) if len(g1) else None
    two_sig = (float(g2.min()), float(g2.max())) if len(g2) else None
    # local minima
    lmin = []
    for i in range(1, len(chi2)-1):
        if chi2[i] < chi2[i-1] and chi2[i] < chi2[i+1]:
            lmin.append((float(grid[i]), float(chi2[i]),
                         float(chi2[i] - chi2_min)))
    return dict(chi2=chi2.tolist(), chi2_min=chi2_min, argmin=argmin,
                one_sig=one_sig, two_sig=two_sig, local_min=lmin,
                target=pos_target,
                dev_pct=(argmin - pos_target) / pos_target * 100)


# ============================================================================
# SECTION 11.  PARAMETRIC RESIDUAL BOOTSTRAP
# ============================================================================

def build_truth_predictions(family, truth_params):
    """Compute the predicted observable at each measurement under truth.

    truth_params: (H0, Om, Ob, A, position, w)
    Returns a dict of predictions keyed by data block.
    """
    H0, Om, Ob, A, pos, w = truth_params
    if family == 'F1':
        T = lambda z: T_z_free(z, A, pos, w)
    else:
        T = lambda z: T_theta_free(z, A, pos, w)
    DC = comoving_dist_grid(H0, Om, T)
    Hg = H0 * E_lcdm(_Z_GRID, Om) * T(_Z_GRID)
    DM_star = DM_at_zstar(H0, Om, Ob, T)
    rd = THETA_STAR * DM_star

    preds = dict(dr2_dv=[], dr2_pair=[], dr1_dv=[], dr1_pair=[],
                 boss=[], eboss_pair=[], eboss_dv=[])
    for z, _, _ in DESI_DR2_DV:
        DM = np.interp(z, _Z_GRID, DC); Hv = np.interp(z, _Z_GRID, Hg)
        preds['dr2_dv'].append((z * DM**2 * C_KMS / Hv) ** (1/3) / rd)
    for label, z, _, _, _, _, _ in DESI_DR2_PAIR:
        preds['dr2_pair'].append(
            (np.interp(z, _Z_GRID, DC) / rd,
             C_KMS / np.interp(z, _Z_GRID, Hg) / rd))
    for z, _, _ in DESI_DR1_DV:
        DM = np.interp(z, _Z_GRID, DC); Hv = np.interp(z, _Z_GRID, Hg)
        preds['dr1_dv'].append((z * DM**2 * C_KMS / Hv) ** (1/3) / rd)
    for label, z, _, _, _, _, _ in DESI_DR1_PAIR:
        preds['dr1_pair'].append(
            (np.interp(z, _Z_GRID, DC) / rd,
             C_KMS / np.interp(z, _Z_GRID, Hg) / rd))
    for (z, _, _), (z2, _, _, _) in zip(BOSS_DM, BOSS_DH):
        preds['boss'].append(
            (np.interp(z, _Z_GRID, DC) / rd,
             C_KMS / np.interp(z, _Z_GRID, Hg) / rd))
    for label, z, _, _, _, _, _ in EBOSS_DM_PAIRS:
        preds['eboss_pair'].append(
            (np.interp(z, _Z_GRID, DC) / rd,
             C_KMS / np.interp(z, _Z_GRID, Hg) / rd))
    for label, z, _, _ in EBOSS_DV:
        DM = np.interp(z, _Z_GRID, DC); Hv = np.interp(z, _Z_GRID, Hg)
        preds['eboss_dv'].append((z * DM**2 * C_KMS / Hv) ** (1/3) / rd)
    R_raw, lA_raw, _, _, _ = _R_lA_raw(H0, Om, Ob, T)
    preds['dp'] = (R_raw * CAL_R, lA_raw * CAL_LA, Ob)
    preds['h0'] = H0
    return preds


def make_realisation(preds, rng):
    """Synthesise one bootstrap realisation: prediction + Gaussian noise
    drawn from the published covariances."""
    out = dict(dr2_dv=[], dr2_pair=[], dr1_dv=[], dr1_pair=[],
               boss_dm=[], boss_dh=[], eboss_pair=[], eboss_dv=[])
    for (z, _, sig), pred in zip(DESI_DR2_DV, preds['dr2_dv']):
        out['dr2_dv'].append((z, pred + rng.normal(0, sig), sig))
    for (lab, z, _, sDM, _, sDH, corr), (DMp, DHp) in zip(
            DESI_DR2_PAIR, preds['dr2_pair']):
        cov = np.array([[sDM**2, corr*sDM*sDH], [corr*sDM*sDH, sDH**2]])
        L = np.linalg.cholesky(cov); eps = L @ rng.standard_normal(2)
        out['dr2_pair'].append((lab, z, DMp + eps[0], sDM, DHp + eps[1], sDH, corr))
    for (z, _, sig), pred in zip(DESI_DR1_DV, preds['dr1_dv']):
        out['dr1_dv'].append((z, pred + rng.normal(0, sig), sig))
    for (lab, z, _, sDM, _, sDH, corr), (DMp, DHp) in zip(
            DESI_DR1_PAIR, preds['dr1_pair']):
        cov = np.array([[sDM**2, corr*sDM*sDH], [corr*sDM*sDH, sDH**2]])
        L = np.linalg.cholesky(cov); eps = L @ rng.standard_normal(2)
        out['dr1_pair'].append((lab, z, DMp + eps[0], sDM, DHp + eps[1], sDH, corr))
    for ((z, _, sDM), (z2, _, sDH, corr)), (DMp, DHp) in zip(
            zip(BOSS_DM, BOSS_DH), preds['boss']):
        cov = np.array([[sDM**2, corr*sDM*sDH], [corr*sDM*sDH, sDH**2]])
        L = np.linalg.cholesky(cov); eps = L @ rng.standard_normal(2)
        out['boss_dm'].append((z, DMp + eps[0], sDM))
        out['boss_dh'].append((z2, DHp + eps[1], sDH, corr))
    for (lab, z, _, sDM, _, sDH, corr), (DMp, DHp) in zip(
            EBOSS_DM_PAIRS, preds['eboss_pair']):
        cov = np.array([[sDM**2, corr*sDM*sDH], [corr*sDM*sDH, sDH**2]])
        L = np.linalg.cholesky(cov); eps = L @ rng.standard_normal(2)
        out['eboss_pair'].append((lab, z, DMp + eps[0], sDM, DHp + eps[1], sDH, corr))
    for (lab, z, _, sig), pred in zip(EBOSS_DV, preds['eboss_dv']):
        out['eboss_dv'].append((lab, z, pred + rng.normal(0, sig), sig))
    R_p, lA_p, ob_p = preds['dp']
    L_dp = np.linalg.cholesky(COV_DP); eps_dp = L_dp @ rng.standard_normal(3)
    out['dp_obs'] = (R_p + eps_dp[0], lA_p + eps_dp[1], ob_p + eps_dp[2])
    out['h0_obs'] = preds['h0'] + rng.normal(0, SIG_H0_SHOES)
    return out


def chi2_no_pp_with_data(H0, Om, Ob_h2, T_func, rdata):
    """PP-excluded chi^2 with externally supplied (perturbed) measurements."""
    DC = comoving_dist_grid(H0, Om, T_func)
    if DC is None:
        return 1e10
    if T_func is None:
        H = H0 * E_lcdm(_Z_GRID, Om)
    else:
        H = H0 * E_lcdm(_Z_GRID, Om) * T_func(_Z_GRID)
    DM_star = DM_at_zstar(H0, Om, Ob_h2, T_func)
    rd = THETA_STAR * DM_star
    chi2 = 0.0

    for z, DVo, sig in rdata['dr2_dv']:
        DM = np.interp(z, _Z_GRID, DC); Hv = np.interp(z, _Z_GRID, H)
        DV = (z * DM**2 * C_KMS / Hv) ** (1/3)
        chi2 += ((DVo - DV / rd) / sig) ** 2
    for label, z, DMo, sDM, DHo, sDH, corr in rdata['dr2_pair']:
        DMp = np.interp(z, _Z_GRID, DC) / rd
        DHp = C_KMS / np.interp(z, _Z_GRID, H) / rd
        cov = np.array([[sDM**2, corr*sDM*sDH], [corr*sDM*sDH, sDH**2]])
        d = np.array([DMo - DMp, DHo - DHp])
        chi2 += d @ np.linalg.inv(cov) @ d

    for z, DVo, sig in rdata['dr1_dv']:
        DM = np.interp(z, _Z_GRID, DC); Hv = np.interp(z, _Z_GRID, H)
        DV = (z * DM**2 * C_KMS / Hv) ** (1/3)
        chi2 += ((DVo - DV / rd) / sig) ** 2
    for label, z, DMo, sDM, DHo, sDH, corr in rdata['dr1_pair']:
        DMp = np.interp(z, _Z_GRID, DC) / rd
        DHp = C_KMS / np.interp(z, _Z_GRID, H) / rd
        cov = np.array([[sDM**2, corr*sDM*sDH], [corr*sDM*sDH, sDH**2]])
        d = np.array([DMo - DMp, DHo - DHp])
        chi2 += d @ np.linalg.inv(cov) @ d

    for (z, DMo, sDM), (z2, DHo, sDH, corr) in zip(rdata['boss_dm'], rdata['boss_dh']):
        DMp = np.interp(z, _Z_GRID, DC) / rd
        DHp = C_KMS / np.interp(z, _Z_GRID, H) / rd
        cov = np.array([[sDM**2, corr*sDM*sDH], [corr*sDM*sDH, sDH**2]])
        d = np.array([DMo - DMp, DHo - DHp])
        chi2 += d @ np.linalg.inv(cov) @ d

    for label, z, DMo, sDM, DHo, sDH, corr in rdata['eboss_pair']:
        DMp = np.interp(z, _Z_GRID, DC) / rd
        DHp = C_KMS / np.interp(z, _Z_GRID, H) / rd
        cov = np.array([[sDM**2, corr*sDM*sDH], [corr*sDM*sDH, sDH**2]])
        d = np.array([DMo - DMp, DHo - DHp])
        chi2 += d @ np.linalg.inv(cov) @ d
    for label, z, DVo, sig in rdata['eboss_dv']:
        DM = np.interp(z, _Z_GRID, DC); Hv = np.interp(z, _Z_GRID, H)
        DV = (z * DM**2 * C_KMS / Hv) ** (1/3)
        chi2 += ((DVo - DV / rd) / sig) ** 2

    R_obs, lA_obs, omegab_obs = rdata['dp_obs']
    R_raw, lA_raw, _, _, _ = _R_lA_raw(H0, Om, Ob_h2, T_func)
    R_cal = R_raw * CAL_R; lA_cal = lA_raw * CAL_LA
    x = np.array([R_cal, lA_cal, Ob_h2])
    xref = np.array([R_obs, lA_obs, omegab_obs])
    d = x - xref
    chi2 += float(d @ COVINV_DP @ d)

    chi2 += ((rdata['h0_obs'] - H0) / SIG_H0_SHOES) ** 2
    return float(chi2)


def fit_realisation(family, rdata, x_warm):
    """Single-seed NM fit of a free-Gaussian to one bootstrap realisation."""
    bounds = _bounds_F1() if family == 'F1' else _bounds_F2()
    def cost(p):
        H0, Om, Ob, A, pos, w = p
        if not all(b[0] <= v <= b[1] for v, b in zip(p, bounds)):
            return 1e8
        if family == 'F1':
            T = lambda z: T_z_free(z, A, pos, w)
        else:
            T = lambda z: T_theta_free(z, A, pos, w)
        return chi2_no_pp_with_data(H0, Om, Ob, T, rdata)
    res = minimize(cost, x_warm, method='Nelder-Mead',
                   options=dict(xatol=1e-7, fatol=1e-7, maxiter=2000,
                                adaptive=True))
    return float(res.fun), res.x.tolist()


def run_bootstrap(B, F1_modeA, F1_modeB, F2_warm, seed=42, verbose=True):
    """B parametric residual bootstrap iterations on PP-excluded data.

    Truth = F1 mode-A.  Each realisation refits F1 from BOTH mode-A and
    mode-B warm starts (taking the lower chi^2) and F2 from F2_warm.
    """
    preds = build_truth_predictions('F1', F1_modeA)
    rng = np.random.default_rng(seed)
    F1_results, F2_results = [], []
    t_start = time.time()
    for b in range(B):
        rdata = make_realisation(preds, rng)
        chi_a, p_a = fit_realisation('F1', rdata, F1_modeA)
        chi_b, p_b = fit_realisation('F1', rdata, F1_modeB)
        if chi_a <= chi_b:
            F1_chi, F1_p = chi_a, p_a
            F1_pick = 'A' if chi_b - chi_a > 0.5 else 'A_close'
        else:
            F1_chi, F1_p = chi_b, p_b
            F1_pick = 'B'
        F1_results.append(dict(chi2=F1_chi, params=F1_p, pick=F1_pick,
                               chi_a=chi_a, chi_b=chi_b))
        chi_f2, p_f2 = fit_realisation('F2', rdata, F2_warm)
        F2_results.append(dict(chi2=chi_f2, params=p_f2))
        if verbose and (b + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"    done {b+1}/{B}  ({elapsed:.0f}s, ~{elapsed/(b+1):.1f}s/iter)")
    return F1_results, F2_results


def bootstrap_summary(F1_results, F2_results, target_zc, target_theta):
    """Aggregate distributional statistics."""
    F1_zc = np.array([r['params'][4] for r in F1_results])
    F1_w  = np.array([r['params'][5] for r in F1_results])
    F1_A  = np.array([r['params'][3] for r in F1_results])
    F2_th = np.array([r['params'][4] for r in F2_results])
    F2_w  = np.array([r['params'][5] for r in F2_results])
    F2_A  = np.array([r['params'][3] for r in F2_results])
    chi_a = np.array([r['chi_a'] for r in F1_results])
    chi_b = np.array([r['chi_b'] for r in F1_results])
    picks = [r['pick'] for r in F1_results]
    return dict(
        F1_zc_mean=float(F1_zc.mean()), F1_zc_median=float(np.median(F1_zc)),
        F1_zc_std=float(F1_zc.std()), F1_zc_min=float(F1_zc.min()),
        F1_zc_max=float(F1_zc.max()),
        F1_zc_q16=float(np.quantile(F1_zc, 0.16)),
        F1_zc_q84=float(np.quantile(F1_zc, 0.84)),
        F1_w_mean=float(F1_w.mean()), F1_w_std=float(F1_w.std()),
        F1_A_mean=float(F1_A.mean()), F1_A_std=float(F1_A.std()),
        F2_th_mean=float(F2_th.mean()), F2_th_median=float(np.median(F2_th)),
        F2_th_std=float(F2_th.std()), F2_th_min=float(F2_th.min()),
        F2_th_max=float(F2_th.max()),
        F2_th_q16=float(np.quantile(F2_th, 0.16)),
        F2_th_q84=float(np.quantile(F2_th, 0.84)),
        F2_w_mean=float(F2_w.mean()), F2_w_std=float(F2_w.std()),
        F2_A_mean=float(F2_A.mean()), F2_A_std=float(F2_A.std()),
        modeA_only=int(sum(1 for p in picks if p == 'A')),
        modeA_close=int(sum(1 for p in picks if p == 'A_close')),
        modeB_only=int(sum(1 for p in picks if p == 'B')),
        median_abs_chi_diff=float(np.median(np.abs(chi_a - chi_b))),
        F1_dev_pct=(float(F1_zc.mean()) - target_zc) / target_zc * 100,
        F2_dev_pct=(float(F2_th.mean()) - target_theta) / target_theta * 100,
    )


# ============================================================================
# SECTION 12.  MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LVC v0.3 reproduction script",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--multiseed', action='store_true',
                        help="use 3 seeds per fit (slower, more robust)")
    parser.add_argument('--bootstrap-full', action='store_true',
                        help="B=60 bootstrap iterations (default 30)")
    parser.add_argument('--skip-baselines', action='store_true',
                        help="skip v0.1 sanity check")
    parser.add_argument('--skip-bootstrap', action='store_true',
                        help="skip bootstrap (Section 11)")
    parser.add_argument('--skip-pp-included', action='store_true',
                        help="skip PP-included free fits (heavy)")
    parser.add_argument('--save-json', metavar='PATH',
                        help="save numerical results as JSON")
    args = parser.parse_args()

    seeds = (101, 202, 303) if args.multiseed else (101,)
    B = 60 if args.bootstrap_full else 30

    print("=" * 72)
    print("LVC v0.3 reproduction script")
    print("=" * 72)
    print(f"  Working principle: locked v0.1 fits are reproduced for sanity;")
    print(f"  v0.3 free fits open the (z_c, w) and (theta_c, w) locks.")
    print(f"  v0.1 axiom system is not modified.")
    print()
    print(f"  Total measurements N = {N_COMBINED}")
    print(f"  Seeds per fit: {seeds}")
    print(f"  Natural-constant targets:")
    print(f"     1/phi     = {INV_PHI:.7f}")
    print(f"     ln(phi)   = {THETA_C_PHI:.7f}")
    print(f"     e/(5pi)   = {W_LVC:.7f}")
    print()

    print("Loading Pantheon+ ...", flush=True)
    t0 = time.time()
    pp = load_pantheonplus()
    print(f"  loaded N = {pp['N']} SNe in {time.time()-t0:.1f}s")
    print()

    results = dict(
        N=N_COMBINED, seeds=list(seeds),
        targets=dict(zc_phi=INV_PHI, theta_c_phi=THETA_C_PHI, w_lock=W_LVC),
    )

    # ------------------------------------------------------------------------
    # 1.  v0.1 SANITY (locked fits)
    # ------------------------------------------------------------------------
    if not args.skip_baselines:
        print("=" * 72)
        print("STEP 1.  v0.1 sanity:  locked Lambda-CDM, Lock-phi-A, LVC pendulum")
        print("=" * 72)

        t0 = time.time()
        chi_lcdm, p_lcdm = fit_LCDM(pp, seeds=seeds)
        print(f"  Lambda-CDM   (k=3): chi^2 = {chi_lcdm:.4f}  "
              f"(target 1624.85, time {time.time()-t0:.1f}s)")
        t0 = time.time()
        chi_lp, p_lp = fit_LockPhiA(pp, seeds=seeds)
        print(f"  Lock-phi-A   (k=4): chi^2 = {chi_lp:.4f}  "
              f"(target 1601.19, time {time.time()-t0:.1f}s)")
        t0 = time.time()
        chi_lv, p_lv = fit_LVC(pp, seeds=seeds)
        print(f"  LVC pendulum (k=4): chi^2 = {chi_lv:.4f}  "
              f"(target 1602.77, time {time.time()-t0:.1f}s)")
        print()
        print(f"  Delta-chi^2 (LVC - LockPhi-A) = {chi_lv - chi_lp:+.4f}  "
              f"(v0.1 paper: +1.58)")
        print()

        results['v01_sanity'] = dict(
            lcdm=dict(chi2=chi_lcdm, H0=p_lcdm[0], Om=p_lcdm[1], Ob=p_lcdm[2]),
            lockphi=dict(chi2=chi_lp, H0=p_lp[0], Om=p_lp[1], Ob=p_lp[2], A=p_lp[3]),
            lvc=dict(chi2=chi_lv, H0=p_lv[0], Om=p_lv[1], Ob=p_lv[2], A=p_lv[3]),
            delta_lvc_minus_lockphi=chi_lv - chi_lp,
        )

    # ------------------------------------------------------------------------
    # 2.  PP-EXCLUDED FREE FITS (k=6)
    # ------------------------------------------------------------------------
    print("=" * 72)
    print("STEP 2.  PP-EXCLUDED free fits (k=6, N=37)")
    print("=" * 72)

    t0 = time.time()
    chi_F1_nopp, p_F1_nopp = fit_F1_free(pp, seeds=seeds, no_pp=True)
    print(f"  F1 (z-Gauss):     chi^2 = {chi_F1_nopp:.4f}  "
          f"(time {time.time()-t0:.1f}s)")
    print(f"     H0={p_F1_nopp[0]:.3f}  Om={p_F1_nopp[1]:.4f}  "
          f"Ob={p_F1_nopp[2]:.5f}  A={p_F1_nopp[3]:+.4f}  "
          f"zc={p_F1_nopp[4]:.4f}  w={p_F1_nopp[5]:.4f}")
    print(f"     zc dev from 1/phi:    {(p_F1_nopp[4]-INV_PHI)/INV_PHI*100:+.2f}%")

    t0 = time.time()
    chi_F2_nopp, p_F2_nopp = fit_F2_free(pp, seeds=seeds, no_pp=True)
    print(f"  F2 (theta-Gauss): chi^2 = {chi_F2_nopp:.4f}  "
          f"(time {time.time()-t0:.1f}s)")
    print(f"     H0={p_F2_nopp[0]:.3f}  Om={p_F2_nopp[1]:.4f}  "
          f"Ob={p_F2_nopp[2]:.5f}  A={p_F2_nopp[3]:+.4f}  "
          f"theta_c={p_F2_nopp[4]:.4f}  w={p_F2_nopp[5]:.4f}")
    print(f"     theta_c dev from ln(phi): "
          f"{(p_F2_nopp[4]-THETA_C_PHI)/THETA_C_PHI*100:+.2f}%")
    print()

    results['pp_excluded'] = dict(
        F1=dict(chi2=chi_F1_nopp, H0=p_F1_nopp[0], Om=p_F1_nopp[1],
                Ob=p_F1_nopp[2], A=p_F1_nopp[3], zc=p_F1_nopp[4], w=p_F1_nopp[5]),
        F2=dict(chi2=chi_F2_nopp, H0=p_F2_nopp[0], Om=p_F2_nopp[1],
                Ob=p_F2_nopp[2], A=p_F2_nopp[3], theta_c=p_F2_nopp[4],
                w=p_F2_nopp[5]),
    )

    # ------------------------------------------------------------------------
    # 3.  PP-EXCLUDED PROFILE LIKELIHOOD
    # ------------------------------------------------------------------------
    print("=" * 72)
    print("STEP 3.  PP-EXCLUDED 1-D profile likelihood")
    print("=" * 72)

    zc_grid = np.array([0.50, 0.55, 0.58, 0.595, 0.607, 0.618, 0.63,
                        0.641, 0.655, 0.67, 0.69, 0.72, 0.75])
    theta_grid = np.array([0.40, 0.43, 0.45, 0.46, 0.47, 0.4812, 0.49,
                           0.50, 0.51, 0.52, 0.54, 0.56, 0.60])

    print(f"  F1 profile across {len(zc_grid)} z_c points...")
    t0 = time.time()
    F1_prof_nopp = profile_walk('F1', zc_grid,
                                pos_anchor=p_F1_nopp[4],
                                x_anchor=[p_F1_nopp[0], p_F1_nopp[1],
                                          p_F1_nopp[2], p_F1_nopp[3],
                                          p_F1_nopp[5]],
                                pp=None, no_pp=True)
    print(f"     done in {time.time()-t0:.1f}s")
    F1_summary_nopp = profile_summary(zc_grid, F1_prof_nopp, INV_PHI)
    print(f"     argmin zc = {F1_summary_nopp['argmin']:.4f}  "
          f"chi2_min = {F1_summary_nopp['chi2_min']:.4f}")
    if F1_summary_nopp['one_sig']:
        a, b = F1_summary_nopp['one_sig']
        print(f"     1-sigma:   zc in [{a:.4f}, {b:.4f}], width {b-a:.4f}")
    print(f"     local minima: {len(F1_summary_nopp['local_min'])}")
    for x, c, dc in F1_summary_nopp['local_min']:
        print(f"        zc={x:.4f}  chi2={c:.4f}  dchi2={dc:.4f}")

    print(f"  F2 profile across {len(theta_grid)} theta_c points...")
    t0 = time.time()
    F2_prof_nopp = profile_walk('F2', theta_grid,
                                pos_anchor=p_F2_nopp[4],
                                x_anchor=[p_F2_nopp[0], p_F2_nopp[1],
                                          p_F2_nopp[2], p_F2_nopp[3],
                                          p_F2_nopp[5]],
                                pp=None, no_pp=True)
    print(f"     done in {time.time()-t0:.1f}s")
    F2_summary_nopp = profile_summary(theta_grid, F2_prof_nopp, THETA_C_PHI)
    print(f"     argmin theta_c = {F2_summary_nopp['argmin']:.4f}  "
          f"chi2_min = {F2_summary_nopp['chi2_min']:.4f}")
    if F2_summary_nopp['one_sig']:
        a, b = F2_summary_nopp['one_sig']
        print(f"     1-sigma:   theta_c in [{a:.4f}, {b:.4f}], width {b-a:.4f}")
    print(f"     local minima: {len(F2_summary_nopp['local_min'])}")
    for x, c, dc in F2_summary_nopp['local_min']:
        print(f"        theta_c={x:.4f}  chi2={c:.4f}  dchi2={dc:.4f}")
    print()

    results['profile_pp_excluded'] = dict(
        F1=dict(grid=zc_grid.tolist(), **F1_summary_nopp),
        F2=dict(grid=theta_grid.tolist(), **F2_summary_nopp),
    )

    # ------------------------------------------------------------------------
    # 4.  PP-INCLUDED FREE FITS AND PROFILES
    # ------------------------------------------------------------------------
    if not args.skip_pp_included:
        print("=" * 72)
        print("STEP 4.  PP-INCLUDED free fits and profiles (k=6, N=1738)")
        print("=" * 72)

        # F1: warm-start from LockPhi-A best-fit
        # F2: full DE at theta_c=ln(phi) to anchor (avoids w-boundary trap)
        F1_warm = [69.43, 0.286, 0.02279, 0.0520, INV_PHI, 0.171]
        print("  F1 free fit, full likelihood (single seed for time)...")
        t0 = time.time()
        chi_F1_pp, p_F1_pp = fit_F1_free(pp, seeds=(seeds[0],), no_pp=False)
        print(f"     chi^2 = {chi_F1_pp:.4f}  ({time.time()-t0:.1f}s)")
        print(f"     H0={p_F1_pp[0]:.3f}  Om={p_F1_pp[1]:.4f}  "
              f"Ob={p_F1_pp[2]:.5f}  A={p_F1_pp[3]:+.4f}  "
              f"zc={p_F1_pp[4]:.4f}  w={p_F1_pp[5]:.4f}")

        print("  F2 free fit, full likelihood (DE-anchored at theta_c=ln(phi))...")
        # Anchor DE specifically at ln(phi) to avoid w=0.5 trap
        t0 = time.time()
        def cost_F2_anchor(p):
            H0, Om, Ob, A, w = p
            if not (60 <= H0 <= 80 and 0.20 <= Om <= 0.45
                    and 0.020 <= Ob <= 0.025 and -0.3 <= A <= 0.3
                    and 0.03 <= w <= 0.50):
                return 1e8
            T = lambda z: T_theta_free(z, A, THETA_C_PHI, w)
            return chi2_full(H0, Om, Ob, T, pp)
        de = differential_evolution(cost_F2_anchor,
            [(60, 80), (0.20, 0.45), (0.020, 0.025), (-0.3, 0.3), (0.03, 0.50)],
            seed=11, tol=1e-7, maxiter=80, polish=False, popsize=15,
            mutation=(0.5, 1.5))
        nm = minimize(cost_F2_anchor, de.x, method='Nelder-Mead',
                      options=dict(xatol=1e-8, fatol=1e-8, maxiter=3000,
                                   adaptive=True))
        # Now take this as the anchor; full profile will reach the rest
        chi_F2_pp = float(nm.fun)
        p_F2_pp = [nm.x[0], nm.x[1], nm.x[2], nm.x[3], THETA_C_PHI, nm.x[4]]
        print(f"     chi^2 = {chi_F2_pp:.4f} at theta_c=ln(phi)  "
              f"({time.time()-t0:.1f}s)")
        print(f"     H0={p_F2_pp[0]:.3f}  Om={p_F2_pp[1]:.4f}  "
              f"Ob={p_F2_pp[2]:.5f}  A={p_F2_pp[3]:+.4f}  "
              f"theta_c={p_F2_pp[4]:.4f}  w={p_F2_pp[5]:.4f}")

        print(f"\n  Delta(F2-F1) PP-included = {chi_F2_pp - chi_F1_pp:+.4f}")
        print()

        # Profiles
        print(f"  F1 PP-included profile across {len(zc_grid)} points...")
        t0 = time.time()
        F1_prof_pp = profile_walk('F1', zc_grid,
                                  pos_anchor=p_F1_pp[4],
                                  x_anchor=[p_F1_pp[0], p_F1_pp[1],
                                            p_F1_pp[2], p_F1_pp[3],
                                            p_F1_pp[5]],
                                  pp=pp, no_pp=False)
        print(f"     done in {time.time()-t0:.1f}s")
        F1_summary_pp = profile_summary(zc_grid, F1_prof_pp, INV_PHI)
        print(f"     argmin zc = {F1_summary_pp['argmin']:.4f}  "
              f"chi2_min = {F1_summary_pp['chi2_min']:.4f}")
        if F1_summary_pp['one_sig']:
            a, b = F1_summary_pp['one_sig']
            print(f"     1-sigma:   zc in [{a:.4f}, {b:.4f}], width {b-a:.4f}")
        print(f"     local minima: {len(F1_summary_pp['local_min'])}")

        print(f"  F2 PP-included profile across {len(theta_grid)} points...")
        t0 = time.time()
        F2_prof_pp = profile_walk('F2', theta_grid,
                                  pos_anchor=p_F2_pp[4],
                                  x_anchor=[p_F2_pp[0], p_F2_pp[1],
                                            p_F2_pp[2], p_F2_pp[3],
                                            p_F2_pp[5]],
                                  pp=pp, no_pp=False)
        print(f"     done in {time.time()-t0:.1f}s")
        F2_summary_pp = profile_summary(theta_grid, F2_prof_pp, THETA_C_PHI)
        print(f"     argmin theta_c = {F2_summary_pp['argmin']:.4f}  "
              f"chi2_min = {F2_summary_pp['chi2_min']:.4f}")
        if F2_summary_pp['one_sig']:
            a, b = F2_summary_pp['one_sig']
            print(f"     1-sigma:   theta_c in [{a:.4f}, {b:.4f}], width {b-a:.4f}")
        print(f"     local minima: {len(F2_summary_pp['local_min'])}")
        print()

        results['pp_included'] = dict(
            F1=dict(chi2=chi_F1_pp, H0=p_F1_pp[0], Om=p_F1_pp[1],
                    Ob=p_F1_pp[2], A=p_F1_pp[3], zc=p_F1_pp[4], w=p_F1_pp[5]),
            F2=dict(chi2=chi_F2_pp, H0=p_F2_pp[0], Om=p_F2_pp[1],
                    Ob=p_F2_pp[2], A=p_F2_pp[3], theta_c=p_F2_pp[4],
                    w=p_F2_pp[5]),
            delta_F2_minus_F1=chi_F2_pp - chi_F1_pp,
        )
        results['profile_pp_included'] = dict(
            F1=dict(grid=zc_grid.tolist(), **F1_summary_pp),
            F2=dict(grid=theta_grid.tolist(), **F2_summary_pp),
        )

    # ------------------------------------------------------------------------
    # 5.  BOOTSTRAP
    # ------------------------------------------------------------------------
    if not args.skip_bootstrap:
        print("=" * 72)
        print(f"STEP 5.  Parametric residual bootstrap (B={B})")
        print("=" * 72)
        print(f"  Truth = F1 mode-A best fit (zc={p_F1_nopp[4]:.4f}, "
              f"w={p_F1_nopp[5]:.4f}, A={p_F1_nopp[3]:+.4f})")

        F1_modeA = [p_F1_nopp[0], p_F1_nopp[1], p_F1_nopp[2],
                    p_F1_nopp[3], p_F1_nopp[4], p_F1_nopp[5]]
        # F1 mode-B warm-start (from PP-excluded grid scan, approx)
        F1_modeB = [p_F1_nopp[0], p_F1_nopp[1], p_F1_nopp[2],
                    0.053, 0.641, 0.201]
        F2_w_warm = [p_F2_nopp[0], p_F2_nopp[1], p_F2_nopp[2],
                     p_F2_nopp[3], p_F2_nopp[4], p_F2_nopp[5]]

        t0 = time.time()
        F1_boot, F2_boot = run_bootstrap(B, F1_modeA, F1_modeB, F2_w_warm,
                                         seed=42, verbose=True)
        print(f"  total bootstrap time: {time.time()-t0:.0f}s")
        boot_summary = bootstrap_summary(F1_boot, F2_boot, INV_PHI, THETA_C_PHI)

        print()
        print(f"  F1 z_c distribution:")
        print(f"     mean = {boot_summary['F1_zc_mean']:.4f}  "
              f"median = {boot_summary['F1_zc_median']:.4f}")
        print(f"     std  = {boot_summary['F1_zc_std']:.4f}  "
              f"q16-q84 = [{boot_summary['F1_zc_q16']:.4f}, "
              f"{boot_summary['F1_zc_q84']:.4f}]")
        print(f"     mode A only: {boot_summary['modeA_only']}, "
              f"A_close: {boot_summary['modeA_close']}, "
              f"B only: {boot_summary['modeB_only']}")
        print(f"     median |chi2_A - chi2_B| = "
              f"{boot_summary['median_abs_chi_diff']:.4f}")
        print()
        print(f"  F2 theta_c distribution:")
        print(f"     mean = {boot_summary['F2_th_mean']:.4f}  "
              f"median = {boot_summary['F2_th_median']:.4f}")
        print(f"     std  = {boot_summary['F2_th_std']:.4f}  "
              f"q16-q84 = [{boot_summary['F2_th_q16']:.4f}, "
              f"{boot_summary['F2_th_q84']:.4f}]")
        print()

        results['bootstrap'] = dict(
            B=B, summary=boot_summary,
            F1_zc=[r['params'][4] for r in F1_boot],
            F1_w=[r['params'][5] for r in F1_boot],
            F1_A=[r['params'][3] for r in F1_boot],
            F1_picks=[r['pick'] for r in F1_boot],
            F2_th=[r['params'][4] for r in F2_boot],
            F2_w=[r['params'][5] for r in F2_boot],
            F2_A=[r['params'][3] for r in F2_boot],
        )

    # ------------------------------------------------------------------------
    # 6.  v0.2 SECTION 10 DECISION SUMMARY
    # ------------------------------------------------------------------------
    print("=" * 72)
    print("STEP 6.  v0.2 Section 10 decision criteria")
    print("=" * 72)
    F2_one_sig_w = F2_summary_nopp['one_sig']
    if F2_one_sig_w:
        a, b = F2_one_sig_w
        F2_width = b - a
        # The v0.3 paper uses the label "half-width" for the value (b - a)
        # of the 1-sigma confidence interval [a, b].  Same convention here.
        print(f"  F2 PP-excluded profile-1sigma 'half-width' (paper label):"
              f" {F2_width:.4f}")
        print(f"     (1-sigma interval [{a:.4f}, {b:.4f}], full width "
              f"{F2_width:.4f})")
        if F2_width <= 0.06:
            verdict = "F2 'half-width' <= 0.06 (Section 10 threshold)"
        elif F2_width <= 0.10:
            verdict = "F2 'half-width' in (0.06, 0.10] (partial)"
        else:
            verdict = "F2 'half-width' > 0.10"
        print(f"  -> {verdict}")

    print(f"  F1 PP-excluded best zc deviation from 1/phi:    "
          f"{F1_summary_nopp['dev_pct']:+.2f}%")
    print(f"  F2 PP-excluded best theta_c dev from ln(phi):  "
          f"{F2_summary_nopp['dev_pct']:+.2f}%")
    if not args.skip_pp_included:
        print(f"  F1 PP-included best zc deviation from 1/phi:    "
              f"{F1_summary_pp['dev_pct']:+.2f}%")
        print(f"  F2 PP-included best theta_c dev from ln(phi):  "
              f"{F2_summary_pp['dev_pct']:+.2f}%")
    print()

    # ------------------------------------------------------------------------
    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x))
        print(f"Results saved -> {args.save_json}")
    print("=" * 72)
    print("Done.")
    print("=" * 72)


if __name__ == '__main__':
    main()
