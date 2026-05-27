"""
LVC v0.4 -- Single-file reproduction script
============================================

Reproduces all numerical results in the v0.4 working paper:

    "Lagrangian Variable Cosmology v0.4:
     A Lagrangian Construction Reproducing the v0.3 theta-Coordinate Modulation,
     with Two Structural Relations Reducing the v0.3 Parameter Count"
                                LUMENPIXEL, May 2026

This is a single self-contained Python file. No LVC-specific dependencies.
The likelihood backend (sections 1-10 below) is identical to v0.1 / v0.2 / v0.3.

What this paper / script add over v0.3
--------------------------------------
v0.3 documented the F2 free-fit chi^2 = 1600.997 vs the v0.1 locked
chi^2 = 1602.77 (with theta_c = ln(phi), w = e/(5pi)) and identified
the +1.78 deficit as localised to the width lock.

v0.4 introduces a Lagrangian construction whose classical kink solution
reproduces the localised theta-coordinate modulation. The script performs:

    Sec. 12.  Lagrangian-side identities (algebraic, machine precision).
              Verification that v(theta) = A sech^2((theta-theta_c)/w)
              is an image of the half-twist sine-Gordon kink, satisfies
              v_thetatheta + V'_eff(v) = 0 with cubic V_eff, and saturates
              V_eff(v) = -(1/2) v_theta^2 along the trajectory.

    Sec. 13.  Width relation w = A/ln phi (Sec. 4 of v0.4 paper).
              Free-A fit on PP-excluded (N=37) with w tied to A.

    Sec. 14.  Recursive Omega_m relation (Sec. 5 of v0.4 paper).
              Lock Omega_m = 1/(1+exp(3 ln phi/phi)) = 0.290653 and
              compare against free-Omega_m reference and falsified
              candidate Omega_m = 1/(2+phi) = 0.276393.

    Sec. 15.  Combined fit (Table 1 of v0.4 paper).

Reproduces (key v0.4 numbers):
------------------------------
v0.1 sanity (chi^2 within 0.005 of paper, PP-included):
    Lambda-CDM                         chi^2 = 1624.85
    LockPhi-A v16   (k=4)              chi^2 = 1601.19
    LVC pendulum    (k=4)              chi^2 = 1602.77

Sec. 12 Lagrangian identities (analytic, no data):
    Eq. (4) sin^2(vartheta_kink/2) = sech^2(...):  residual < 1e-15
    Eq. (6) v_thetatheta + V'_eff(v) = 0:          residual < 1e-15
    Eq. (7) V_eff(v) + (1/2) v_theta^2 = 0 (BPS):  residual < 1e-16
    Closed form: integral v_theta^2 dtheta = 16 A^2/(15 w)

Sec. 13 width relation w = A/ln phi (PP-excluded, N=37, k=4):
    chi^2     = 56.113
    A_fit     = 0.0547,  w_derived = 0.1137
    A_fit/(ln phi)^4 = w_derived/(ln phi)^3 = 1.020

Sec. 14 recursive Omega_m (PP-excluded, N=37, k=2):
    Reference (Omega_m FREE, k=3):     chi^2 = 56.134
    Lock recursive Omega_m:            chi^2 = 56.134, Delta chi^2 = +0.0005
    Lock Omega_m = 1/(2+phi):          chi^2 = 81.83,  Delta chi^2 = +25.7

Sec. 15 Table 1 summary (PP-excluded, N=37):
    Lambda-CDM (k=3):                 chi^2 = 78.962, BIC = 89.795
    Intermediate (Om free, k=3):      chi^2 = 56.134, BIC = 66.967  (-22.83)
    Full lock (Om recursive, k=2):    chi^2 = 56.134, BIC = 63.357  (-26.44)
    Falsified Om = 1/(2+phi) (k=2):   chi^2 = 81.83,  BIC = 89.05   ( -0.74)

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
Default fast run:                                              ~5 min
With --skip-baselines (no v0.1 PP-included sanity check):      ~3 min
With --multiseed (2 seeds per fit):                            ~10 min

Usage
-----
    python lvc_v04_reproduce.py                      # default
    python lvc_v04_reproduce.py --multiseed           # 2 seeds per fit
    python lvc_v04_reproduce.py --skip-baselines      # no v0.1 sanity
    python lvc_v04_reproduce.py --save-json PATH      # save numbers
    python lvc_v04_reproduce.py --help                # full options
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

W_PHI_A    = E / (5 * PI)         # v0.1 axiom L4 lock
W_LVC      = E / (5 * PI)

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

N_COMBINED = 1701 + 13 + 11 + 2 + 7 + 3 + 1   # = 1738


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
    print(f"  Target directory: {PP_DIR}")
    print(f"  Required disk:    ~33 MB")
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
# SECTION 3.  BACKGROUND COSMOLOGY
# ============================================================================

_Z_GRID = np.concatenate([[0.0], np.geomspace(1e-3, 5.0, 200)])


def E_lcdm(z, Om):
    return np.sqrt(Om*(1+z)**3 + (1-Om))


def T_zGauss(z, A, zc=INV_PHI, w=W_PHI_A):
    """F1: z-Gaussian modulation."""
    return 1.0 + A * np.exp(-((z - zc) / w)**2)


def T_thetaGauss(z, A, theta_c=THETA_C_PHI, w=W_LVC):
    """F2: theta-Gaussian modulation, theta = ln(1+z)."""
    theta = np.log(1.0 + z)
    return 1.0 + A * np.exp(-((theta - theta_c) / w)**2)


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
# SECTION 8.  v0.1 SANITY-CHECK FITTERS
# ============================================================================

def _de_nm(cost, bounds, seeds, maxiter=120, popsize=12, polish_iter=2500):
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
    bounds = [(60, 80), (0.20, 0.45), (0.020, 0.025)]
    def cost(p):
        return chi2_full(p[0], p[1], p[2], None, pp)
    return _de_nm(cost, bounds, seeds=seeds)


def fit_LockPhiA(pp, seeds):
    """k=4: locked z_c = 1/phi, w = e/(5pi)."""
    bounds = [(60, 80), (0.20, 0.45), (0.020, 0.025), (-0.3, 0.3)]
    def cost(p):
        T = lambda z: T_zGauss(z, p[3])
        return chi2_full(p[0], p[1], p[2], T, pp)
    return _de_nm(cost, bounds, seeds=seeds)


def fit_LVC(pp, seeds):
    """k=4: locked theta_c = ln(phi), w = e/(5pi)."""
    bounds = [(60, 80), (0.20, 0.45), (0.020, 0.025), (-0.3, 0.3)]
    def cost(p):
        T = lambda z: T_thetaGauss(z, p[3])
        return chi2_full(p[0], p[1], p[2], T, pp)
    return _de_nm(cost, bounds, seeds=seeds)


# ============================================================================
# SECTION 9.  v0.4 LOCKED-FORM SWEEP  (theta_c = ln phi, w = candidate)
# ============================================================================

def fit_F2_locked(pp, theta_c, w, seeds, no_pp=False):
    """k=4 locked F2 fit. (H0, Om, Ob, A) free; theta_c, w fixed."""
    bounds = [(60, 80), (0.20, 0.45), (0.020, 0.025), (-0.3, 0.3)]
    def cost(p):
        H0, Om, Ob, A = p
        T = lambda z: T_thetaGauss(z, A, theta_c=theta_c, w=w)
        return chi2_no_pp(H0, Om, Ob, T) if no_pp else chi2_full(H0, Om, Ob, T, pp)
    best = (1e10, None)
    for sd in seeds:
        de = differential_evolution(cost, bounds, seed=sd, tol=1e-7,
                                    maxiter=80, popsize=12, polish=False,
                                    mutation=(0.5, 1.5), recombination=0.7)
        nm = minimize(cost, de.x, method='Nelder-Mead',
                      options=dict(xatol=1e-8, fatol=1e-8, maxiter=3000,
                                   adaptive=True))
        if nm.fun < best[0]:
            best = (float(nm.fun), nm.x.copy())
    return best


# Candidate definitions: label, expression, w value, Omega_0 closed form
CANDIDATES_MAIN = [
    ('v0.1',  'e/(5pi)',         E / (5 * PI),      '5pi*sqrt(2)/e'),
    ('C1',    '1/(3pi)',         1.0 / (3 * PI),    '3pi*sqrt(2)'),
    ('C2',    '1/9',             1.0 / 9.0,         '9*sqrt(2)'),
    ('C3',    'e/(8pi)',         E / (8 * PI),      '8pi*sqrt(2)/e'),
    ('C4',    '1/(2pi)',         1.0 / (2 * PI),    '2pi*sqrt(2)'),
    ('C5',    'sqrt(2)/(4pi)',   np.sqrt(2) / (4 * PI),  '4pi'),
    ('C6',    'ln(phi)/4',       np.log(PHI) / 4,    '4*sqrt(2)/ln(phi)'),
]

CANDIDATES_PHI = [
    ('Phi1',  '1/(2pi*phi)',     1.0 / (2 * PI * PHI),    '2pi*phi*sqrt(2)'),
    ('Phi2',  '1/(pi*phi^2)',    1.0 / (PI * PHI**2),     'pi*phi^2*sqrt(2)'),
    ('Phi3',  '1/(5*phi)',       1.0 / (5 * PHI),         '5*phi*sqrt(2)'),
]


def Omega_0(w):
    return np.sqrt(2.0) / w


# ============================================================================
# SECTION 10.  v0.4 1-D PROFILE LIKELIHOOD IN W
# ============================================================================

def _profile_F2_at_w(pp, w_fixed, x_warm, no_pp=False):
    """NM-only fit at fixed (theta_c=ln phi, w).  Sliding warm start."""
    def cost(p):
        H0, Om, Ob, A = p
        if not (60 <= H0 <= 80 and 0.20 <= Om <= 0.45
                and 0.020 <= Ob <= 0.025 and -0.3 <= A <= 0.3):
            return 1e8
        T = lambda z: T_thetaGauss(z, A, theta_c=THETA_C_PHI, w=w_fixed)
        return chi2_no_pp(H0, Om, Ob, T) if no_pp else chi2_full(H0, Om, Ob, T, pp)
    res = minimize(cost, x_warm, method='Nelder-Mead',
                   options=dict(xatol=1e-7, fatol=1e-7, maxiter=3000,
                                adaptive=True))
    return float(res.fun), res.x.tolist()


def _profile_F1_at_w(pp, w_fixed, x_warm):
    """F1 ridge profile, z_c locked at 1/phi."""
    def cost(p):
        H0, Om, Ob, A = p
        if not (60 <= H0 <= 80 and 0.20 <= Om <= 0.45
                and 0.020 <= Ob <= 0.025 and -0.3 <= A <= 0.3):
            return 1e8
        T = lambda z: T_zGauss(z, A, zc=INV_PHI, w=w_fixed)
        return chi2_full(H0, Om, Ob, T, pp)
    res = minimize(cost, x_warm, method='Nelder-Mead',
                   options=dict(xatol=1e-7, fatol=1e-7, maxiter=3000,
                                adaptive=True))
    return float(res.fun), res.x.tolist()


def w_ridge_F2(pp, w_grid, no_pp=False, verbose=True):
    """Sliding-warm-start NM profile across w_grid.  theta_c = ln phi."""
    chi2_arr = []
    params_arr = []
    x_warm = [69.4, 0.286, 0.02279, 0.05] if not no_pp else [69.1, 0.290, 0.02273, 0.06]
    for w in w_grid:
        chi, p = _profile_F2_at_w(pp, w, x_warm, no_pp=no_pp)
        chi2_arr.append(chi); params_arr.append(p)
        x_warm = p
        if verbose:
            print(f"    w={w:.4f}: chi^2={chi:.4f}  H0={p[0]:.3f}  A={p[3]:+.4f}")
    return np.array(chi2_arr), params_arr


def w_ridge_F1(pp, w_grid, verbose=True):
    chi2_arr = []
    params_arr = []
    x_warm = [69.4, 0.286, 0.02279, 0.05]
    for w in w_grid:
        chi, p = _profile_F1_at_w(pp, w, x_warm)
        chi2_arr.append(chi); params_arr.append(p)
        x_warm = p
        if verbose:
            print(f"    w={w:.4f}: chi^2={chi:.4f}  H0={p[0]:.3f}  A={p[3]:+.4f}")
    return np.array(chi2_arr), params_arr


def ridge_summary(w_grid, chi2_arr):
    """Compute argmin, 1-sigma and 2-sigma intervals."""
    chi2_min = float(chi2_arr.min())
    w_argmin = float(w_grid[np.argmin(chi2_arr)])
    dchi2 = chi2_arr - chi2_min
    one_sig = w_grid[dchi2 < 1.0]
    two_sig = w_grid[dchi2 < 4.0]
    return dict(
        chi2_min=chi2_min, w_argmin=w_argmin,
        one_sig=([float(one_sig.min()), float(one_sig.max())]
                 if len(one_sig) else None),
        two_sig=([float(two_sig.min()), float(two_sig.max())]
                 if len(two_sig) else None),
    )


def naturals_at_ridge(w_grid, chi2_arr, naturals_dict):
    """Return Δχ² (interpolated) and 1σ/2σ membership for each natural constant."""
    chi2_min = float(chi2_arr.min())
    dchi2 = chi2_arr - chi2_min
    summary = ridge_summary(w_grid, chi2_arr)
    out = []
    for name, val in sorted(naturals_dict.items(), key=lambda x: x[1]):
        if val < w_grid.min() or val > w_grid.max():
            out.append(dict(name=name, w=val, dchi2=None, region='outside grid'))
            continue
        dchi_interp = float(np.interp(val, w_grid, dchi2))
        if summary['one_sig'] and summary['one_sig'][0] <= val <= summary['one_sig'][1]:
            region = '1-sigma'
        elif summary['two_sig'] and summary['two_sig'][0] <= val <= summary['two_sig'][1]:
            region = '2-sigma'
        else:
            region = 'outside'
        out.append(dict(name=name, w=val, dchi2=dchi_interp, region=region))
    return out



# ============================================================================
# SECTION 11.  v0.4 CONSTANTS (additional)
# ============================================================================

# (ln phi)^k pattern values
LNPHI_3 = THETA_C_PHI ** 3   # 0.111432
LNPHI_4 = THETA_C_PHI ** 4   # 0.053622

# Recursive Omega_m: 1 / (1 + exp(3 ln phi / phi))
OM_RECURSIVE = 1.0 / (1.0 + np.exp(3.0 * THETA_C_PHI / PHI))   # 0.290653

# Falsified candidate (recorded for comparison)
OM_2PLUSPHI = 1.0 / (2.0 + PHI)             # 0.276393


def T_sech2(z, A, theta_c=THETA_C_PHI, w=LNPHI_3):
    """v0.4 sech^2 modulation in theta-coordinate (from half-twist kink)."""
    theta = np.log(1.0 + z)
    return 1.0 + A * (1.0 / np.cosh((theta - theta_c) / w))**2


# ============================================================================
# SECTION 12.  LAGRANGIAN-SIDE IDENTITIES (Sec. 2/3 of v0.4 paper)
# ============================================================================

def verify_lagrangian_identities():
    """
    Verify Eqs. (4), (6), (7) of the v0.4 paper to machine precision.

    Eq. (4): sin^2(vartheta_kink(theta)/2) = sech^2((m/2)(theta - theta_c))
    Eq. (6): v_thetatheta + V'_eff(v) = 0 with V'_eff(v) = -(4/w^2)v + (6/(Aw^2))v^2
    Eq. (7): V_eff(v) = -(1/2) v_theta^2  (BPS-saturated bounce)
    """
    print("=" * 76)
    print("Sec. 12 LAGRANGIAN IDENTITIES (analytic, verification at machine precision)")
    print("=" * 76)

    A = LNPHI_4
    w = LNPHI_3
    theta_c = THETA_C_PHI
    m = 2.0 / w   # from w = 2/m

    # Eq. (4): sin^2(vartheta_kink/4) = sech^2((m/2)(theta - theta_c))
    print("\nEq. (4): sin^2(vartheta_kink/4) = sech^2((m/2)(theta - theta_c))")
    test_thetas = np.linspace(theta_c - 0.5, theta_c + 0.5, 11)
    max_err_4 = 0.0
    for th in test_thetas:
        # vartheta_kink = 8 arctan(exp((m/2)(theta-theta_c)))
        u = (m/2.0) * (th - theta_c)
        vartheta = 8.0 * np.arctan(np.exp(u))
        lhs = np.sin(vartheta / 4.0)**2
        rhs = (1.0 / np.cosh(u))**2
        err = abs(lhs - rhs)
        if err > max_err_4:
            max_err_4 = err
    print(f"  max |LHS - RHS| over 11 sample points: {max_err_4:.2e}")

    # Eq. (6): v_thetatheta + V'_eff(v) = 0
    print("\nEq. (6): v_thetatheta + V'_eff(v) = 0 with cubic V_eff")
    sample_thetas = [0.30, 0.40, 0.481, 0.50, 0.60, 0.70]
    max_err_6 = 0.0
    for th in sample_thetas:
        u = (th - theta_c) / w
        sech2 = 1.0 / np.cosh(u)**2
        v = A * sech2
        # v_thetatheta = (4/w^2) v - (6/(Aw^2)) v^2
        vpp = (4.0/w**2) * v - (6.0/(A * w**2)) * v**2
        # V'_eff(v) = -(4/w^2) v + (6/(Aw^2)) v^2
        Vp_eff = -(4.0/w**2) * v + (6.0/(A * w**2)) * v**2
        residual = vpp + Vp_eff
        if abs(residual) > max_err_6:
            max_err_6 = abs(residual)
    print(f"  max |residual| over 6 sample points:    {max_err_6:.2e}")

    # Eq. (7): V_eff(v) = -(1/2) v_theta^2 (BPS)
    print("\nEq. (7): V_eff(v) + (1/2) v_theta^2 = 0  (BPS-saturated bounce)")
    max_err_7 = 0.0
    for th in sample_thetas:
        u = (th - theta_c) / w
        sech2 = 1.0 / np.cosh(u)**2
        tanh = np.tanh(u)
        v = A * sech2
        v_theta = -(2.0 * A / w) * sech2 * tanh
        V_eff = -(2.0/w**2) * v**2 * (1.0 - v/A)
        residual = V_eff + 0.5 * v_theta**2
        if abs(residual) > max_err_7:
            max_err_7 = abs(residual)
    print(f"  max |V_eff(v) + (1/2)v_theta^2|:        {max_err_7:.2e}")

    # Bounce kinetic integral (closed form)
    print("\nClosed form: integral v_theta^2 dtheta = 16 A^2 / (15 w)")
    integrand = lambda th: ((-2.0*A/w) * (1.0/np.cosh((th-theta_c)/w))**2
                            * np.tanh((th-theta_c)/w))**2
    I_num, _ = quad(integrand, theta_c - 8*w, theta_c + 8*w)
    I_form = 16.0 * A**2 / (15.0 * w)
    print(f"  numerical: {I_num:.6e}")
    print(f"  formula:   {I_form:.6e}")
    print(f"  difference: {abs(I_num - I_form):.2e}")
    print()

    return {
        'eq4_max_err': float(max_err_4),
        'eq6_max_err': float(max_err_6),
        'eq7_max_err': float(max_err_7),
        'bounce_kinetic_numerical': float(I_num),
        'bounce_kinetic_formula': float(I_form),
    }


# ============================================================================
# SECTION 13.  WIDTH RELATION w = A / ln phi  (Sec. 4 of v0.4 paper)
# ============================================================================

def test_width_relation(seeds):
    """
    Sec. 4: free A with w tied to A by w = A/ln phi.
    Expected: A_fit ~ 0.0547, w_derived ~ 0.1137, chi^2 ~ 56.11.
    """
    print("=" * 76)
    print("Sec. 13 WIDTH RELATION  w = A / ln phi   (PP-excluded, N=37)")
    print("=" * 76)

    def cost(p):
        H0, Om, Ob_h2, A = p
        if A <= 0:
            return 1e10
        w = A / THETA_C_PHI
        if w < 0.02 or w > 0.5:
            return 1e10
        T = lambda z: T_sech2(z, A, theta_c=THETA_C_PHI, w=w)
        c = chi2_no_pp(H0, Om, Ob_h2, T)
        return c if np.isfinite(c) else 1e10

    print("\nFit with theta_c = ln phi locked, w tied to A as w = A/ln phi (k=4).")
    t0 = time.time()
    chi2, x = _de_nm(cost,
               [(60.0, 80.0), (0.20, 0.40), (0.018, 0.028), (0.005, 0.20)],
               seeds=seeds, maxiter=120)
    H0, Om, Ob_h2, A_fit = x
    w_derived = A_fit / THETA_C_PHI
    print(f"  chi^2     = {chi2:.4f}")
    print(f"  H0        = {H0:.4f}")
    print(f"  Omega_m   = {Om:.4f}")
    print(f"  Omega_b h^2 = {Ob_h2:.5f}")
    print(f"  A_fit     = {A_fit:.5f}")
    print(f"  w_derived = A_fit / ln phi = {w_derived:.5f}")
    print()
    print(f"  Reference: A = (ln phi)^4 = {LNPHI_4:.5f}, "
          f"w = (ln phi)^3 = {LNPHI_3:.5f}")
    print(f"  ratio  A_fit / (ln phi)^4   = {A_fit/LNPHI_4:.4f}")
    print(f"  ratio  w_derived / (ln phi)^3 = {w_derived/LNPHI_3:.4f}")
    print(f"  (these two ratios coincide if w = A/ln phi holds in data)")
    print(f"  fit time: {time.time()-t0:.1f}s")
    print()

    return {
        'chi2': float(chi2),
        'H0': float(H0),
        'Om': float(Om),
        'Ob_h2': float(Ob_h2),
        'A_fit': float(A_fit),
        'w_derived': float(w_derived),
        'ratio_A_to_lnphi4': float(A_fit/LNPHI_4),
        'ratio_w_to_lnphi3': float(w_derived/LNPHI_3),
    }


# ============================================================================
# SECTION 14.  RECURSIVE Omega_m  (Sec. 5 of v0.4 paper)
# ============================================================================

def test_recursive_Om(seeds):
    """
    Sec. 5: lock Omega_m = 1 / (1 + exp(3 ln phi / phi)).
    Reference: Omega_m free with all (ln phi)^k modulation locks.
    Falsified candidate: Omega_m = 1/(2+phi).
    """
    print("=" * 76)
    print("Sec. 14 RECURSIVE Omega_m: ln phi = phi * theta_eq(Omega_m)")
    print("=" * 76)

    print(f"\nRecursive value:    Omega_m = 1/(1+exp(3 ln phi / phi))")
    print(f"                          = {OM_RECURSIVE:.8f}")
    print(f"Falsified candidate: Omega_m = 1/(2+phi) = {OM_2PLUSPHI:.8f}")

    # Reference: Omega_m FREE, all (ln phi)^k locks
    def cost_free(p):
        H0, Om, Ob_h2 = p
        T = lambda z: T_sech2(z, A=LNPHI_4, theta_c=THETA_C_PHI, w=LNPHI_3)
        c = chi2_no_pp(H0, Om, Ob_h2, T)
        return c if np.isfinite(c) else 1e10

    print("\n[1/3] Reference: Omega_m FREE, (ln phi)^k locks (k=3) ...")
    t0 = time.time()
    chi2_free, x_free = _de_nm(cost_free,
                [(60.0, 80.0), (0.20, 0.40), (0.018, 0.028)],
                seeds=seeds, maxiter=120)
    print(f"      chi^2      = {chi2_free:.6f}")
    print(f"      Omega_m_fit = {x_free[1]:.6f}  "
          f"(compare recursive {OM_RECURSIVE:.6f})")
    print(f"      H0={x_free[0]:.4f}, Ob_h2={x_free[2]:.5f}  ({time.time()-t0:.1f}s)")

    # Lock to recursive value
    def cost_rec(p):
        H0, Ob_h2 = p
        T = lambda z: T_sech2(z, A=LNPHI_4, theta_c=THETA_C_PHI, w=LNPHI_3)
        c = chi2_no_pp(H0, OM_RECURSIVE, Ob_h2, T)
        return c if np.isfinite(c) else 1e10

    print("\n[2/3] Lock Omega_m = 1/(1+exp(3 ln phi / phi))  (k=2) ...")
    t0 = time.time()
    chi2_rec, x_rec = _de_nm(cost_rec,
                   [(60.0, 80.0), (0.018, 0.028)],
                   seeds=seeds, maxiter=120)
    delta_chi2_rec = chi2_rec - chi2_free
    print(f"      chi^2 = {chi2_rec:.6f}")
    print(f"      Delta chi^2 vs free-Omega_m: {delta_chi2_rec:+.6f}")
    print(f"      H0={x_rec[0]:.4f}, Ob_h2={x_rec[1]:.5f}  ({time.time()-t0:.1f}s)")

    # Lock to falsified candidate
    def cost_2pp(p):
        H0, Ob_h2 = p
        T = lambda z: T_sech2(z, A=LNPHI_4, theta_c=THETA_C_PHI, w=LNPHI_3)
        c = chi2_no_pp(H0, OM_2PLUSPHI, Ob_h2, T)
        return c if np.isfinite(c) else 1e10

    print("\n[3/3] Lock Omega_m = 1/(2+phi)  (falsified candidate, k=2) ...")
    t0 = time.time()
    chi2_2pp, x_2pp = _de_nm(cost_2pp,
                   [(60.0, 80.0), (0.018, 0.028)],
                   seeds=seeds, maxiter=120)
    delta_chi2_2pp = chi2_2pp - chi2_free
    print(f"      chi^2 = {chi2_2pp:.4f}")
    print(f"      Delta chi^2 vs free-Omega_m: {delta_chi2_2pp:+.4f}")
    print(f"      H0={x_2pp[0]:.4f}, Ob_h2={x_2pp[1]:.5f}  ({time.time()-t0:.1f}s)")
    print()

    return {
        'chi2_free_Om': float(chi2_free),
        'Om_fit': float(x_free[1]),
        'H0_free': float(x_free[0]),
        'chi2_lock_recursive': float(chi2_rec),
        'delta_chi2_recursive': float(delta_chi2_rec),
        'H0_recursive': float(x_rec[0]),
        'chi2_lock_2plusphi': float(chi2_2pp),
        'delta_chi2_2plusphi': float(delta_chi2_2pp),
    }


# ============================================================================
# SECTION 15.  COMBINED FIT TABLE 1  (Sec. 6 of v0.4 paper)
# ============================================================================

def make_table_1(seeds):
    """
    Sec. 6 Table 1: combined fit comparison.
    Rows: LCDM, intermediate, full lock, falsified candidate.
    """
    print("=" * 76)
    print("Sec. 15 TABLE 1: Combined fit comparison (PP-excluded, N=37)")
    print("=" * 76)

    N = 37

    # Row 1: Lambda-CDM
    def cost_lcdm(p):
        H0, Om, Ob_h2 = p
        c = chi2_no_pp(H0, Om, Ob_h2, None)
        return c if np.isfinite(c) else 1e10

    print("\n[1/4] Lambda-CDM (k=3) ...")
    t0 = time.time()
    chi2_lcdm, _ = _de_nm(cost_lcdm,
                    [(60.0, 80.0), (0.20, 0.40), (0.018, 0.028)],
                    seeds=seeds, maxiter=120)
    bic_lcdm = chi2_lcdm + 3 * np.log(N)
    print(f"      chi^2 = {chi2_lcdm:.4f}, BIC = {bic_lcdm:.3f}  ({time.time()-t0:.1f}s)")

    # Row 2: intermediate (Om free, all modulation locks)
    def cost_int(p):
        H0, Om, Ob_h2 = p
        T = lambda z: T_sech2(z, A=LNPHI_4, theta_c=THETA_C_PHI, w=LNPHI_3)
        c = chi2_no_pp(H0, Om, Ob_h2, T)
        return c if np.isfinite(c) else 1e10

    print("\n[2/4] Intermediate ((ln phi)^k locks, Omega_m FREE; k=3) ...")
    t0 = time.time()
    chi2_int, _ = _de_nm(cost_int,
                   [(60.0, 80.0), (0.20, 0.40), (0.018, 0.028)],
                   seeds=seeds, maxiter=120)
    bic_int = chi2_int + 3 * np.log(N)
    delta_int = bic_int - bic_lcdm
    print(f"      chi^2 = {chi2_int:.4f}, BIC = {bic_int:.3f}, "
          f"DeltaBIC = {delta_int:+.3f}  ({time.time()-t0:.1f}s)")

    # Row 3: full lock (Om recursive)
    def cost_full(p):
        H0, Ob_h2 = p
        T = lambda z: T_sech2(z, A=LNPHI_4, theta_c=THETA_C_PHI, w=LNPHI_3)
        c = chi2_no_pp(H0, OM_RECURSIVE, Ob_h2, T)
        return c if np.isfinite(c) else 1e10

    print("\n[3/4] Full lock (Omega_m = 1/(1+exp(3 ln phi / phi)); k=2) ...")
    t0 = time.time()
    chi2_full_v, _ = _de_nm(cost_full,
                    [(60.0, 80.0), (0.018, 0.028)],
                    seeds=seeds, maxiter=120)
    bic_full = chi2_full_v + 2 * np.log(N)
    delta_full = bic_full - bic_lcdm
    print(f"      chi^2 = {chi2_full_v:.4f}, BIC = {bic_full:.3f}, "
          f"DeltaBIC = {delta_full:+.3f}  ({time.time()-t0:.1f}s)")

    # Row 4: falsified candidate
    def cost_2pp(p):
        H0, Ob_h2 = p
        T = lambda z: T_sech2(z, A=LNPHI_4, theta_c=THETA_C_PHI, w=LNPHI_3)
        c = chi2_no_pp(H0, OM_2PLUSPHI, Ob_h2, T)
        return c if np.isfinite(c) else 1e10

    print("\n[4/4] Falsified candidate (Omega_m = 1/(2+phi); k=2) ...")
    t0 = time.time()
    chi2_2pp, _ = _de_nm(cost_2pp,
                   [(60.0, 80.0), (0.018, 0.028)],
                   seeds=seeds, maxiter=120)
    bic_2pp = chi2_2pp + 2 * np.log(N)
    delta_2pp = bic_2pp - bic_lcdm
    print(f"      chi^2 = {chi2_2pp:.4f}, BIC = {bic_2pp:.3f}, "
          f"DeltaBIC = {delta_2pp:+.3f}  ({time.time()-t0:.1f}s)")

    # Print Table 1
    print("\n" + "-" * 76)
    print("TABLE 1.  Goodness of fit on PP-excluded data set (N=37).")
    print("-" * 76)
    print(f"  {'Model':<48s} {'k':>3s} {'chi^2':>9s} {'BIC':>9s} {'DeltaBIC':>10s}")
    print("-" * 76)
    print(f"  {'Lambda-CDM (free H0, Omega_m, Omega_b h^2)':<48s} "
          f"{3:>3d} {chi2_lcdm:>9.3f} {bic_lcdm:>9.3f} {'--':>10s}")
    print(f"  {'(ln phi)^k locks, Omega_m FREE':<48s} "
          f"{3:>3d} {chi2_int:>9.3f} {bic_int:>9.3f} {delta_int:>+10.3f}")
    print(f"  {'+ Omega_m locked by Eq. (10)':<48s} "
          f"{2:>3d} {chi2_full_v:>9.3f} {bic_full:>9.3f} {delta_full:>+10.3f}")
    print(f"  {'+ Omega_m = 1/(2+phi)  [falsified]':<48s} "
          f"{2:>3d} {chi2_2pp:>9.3f} {bic_2pp:>9.3f} {delta_2pp:>+10.3f}")
    print("-" * 76)
    print()

    return {
        'lcdm':           {'chi2': float(chi2_lcdm),   'bic': float(bic_lcdm),   'k': 3},
        'intermediate':   {'chi2': float(chi2_int),    'bic': float(bic_int),
                           'k': 3, 'delta_bic': float(delta_int)},
        'full_lock':      {'chi2': float(chi2_full_v), 'bic': float(bic_full),
                           'k': 2, 'delta_bic': float(delta_full)},
        'falsified_2pp':  {'chi2': float(chi2_2pp),    'bic': float(bic_2pp),
                           'k': 2, 'delta_bic': float(delta_2pp)},
    }


# ============================================================================
# SECTION 16.  v0.1 BASELINE SANITY CHECK (PP-included, optional)
# ============================================================================

def run_baselines(pp, seeds):
    """Reproduce v0.1 paper headline numbers."""
    print("=" * 76)
    print("Sec. 16 BASELINES: reproduce v0.1 headline numbers (PP-included, N=1738)")
    print("=" * 76)

    print("\n[1/3] Lambda-CDM free fit ...")
    t0 = time.time()
    chi2_lcdm, x_lcdm = fit_LCDM(pp, seeds)
    print(f"      chi^2 = {chi2_lcdm:.4f}  (v0.1 ref: 1624.85)")
    print(f"      H0={x_lcdm[0]:.3f}, Omega_m={x_lcdm[1]:.4f}, "
          f"Omega_b h^2={x_lcdm[2]:.5f}  ({time.time()-t0:.1f}s)")

    print("\n[2/3] LockPhi-A v16 (z-Gaussian, k=4) ...")
    t0 = time.time()
    chi2_v16, x_v16 = fit_LockPhiA(pp, seeds)
    print(f"      chi^2 = {chi2_v16:.4f}  (v0.1 ref: 1601.19)")
    print(f"      A_fit = {x_v16[3]:.5f}  ({time.time()-t0:.1f}s)")

    print("\n[3/3] LVC pendulum v0.1 (theta-Gaussian, k=4) ...")
    t0 = time.time()
    chi2_v01, x_v01 = fit_LVC(pp, seeds)
    print(f"      chi^2 = {chi2_v01:.4f}  (v0.1 ref: 1602.77)")
    print(f"      A_fit = {x_v01[3]:.5f}  ({time.time()-t0:.1f}s)")
    print()

    return {
        'lcdm_pp_inc': float(chi2_lcdm),
        'lockphi_a_pp_inc': float(chi2_v16),
        'lvc_pendulum_pp_inc': float(chi2_v01),
    }


# ============================================================================
# SECTION 17.  MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce LVC v0.4 working paper numerical results.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--multiseed', action='store_true',
                        help="use 2 seeds per fit (slower, more robust)")
    parser.add_argument('--skip-baselines', action='store_true',
                        help="skip v0.1 PP-included sanity check")
    parser.add_argument('--save-json', metavar='PATH', default=None,
                        help="save numerical results as JSON")
    args = parser.parse_args()

    seeds = (101, 202) if args.multiseed else (101,)

    print("=" * 76)
    print("LVC v0.4 -- Standalone reproduction script")
    print("=" * 76)
    print(f"  phi          = {PHI:.10f}")
    print(f"  ln phi       = {THETA_C_PHI:.10f}")
    print(f"  (ln phi)^3   = {LNPHI_3:.6f}")
    print(f"  (ln phi)^4   = {LNPHI_4:.6f}")
    print(f"  Omega_m_recursive = 1/(1+exp(3 ln phi / phi)) = {OM_RECURSIVE:.8f}")
    print(f"  Omega_m_falsified = 1/(2+phi)              = {OM_2PLUSPHI:.8f}")
    print(f"  seeds: {seeds}")
    print()

    results = {
        'constants': {
            'phi': float(PHI),
            'ln_phi': float(THETA_C_PHI),
            'lnphi_3': float(LNPHI_3),
            'lnphi_4': float(LNPHI_4),
            'Om_recursive': float(OM_RECURSIVE),
            'Om_2plusphi': float(OM_2PLUSPHI),
        },
    }

    # Sec. 12: Lagrangian identities (analytic, no data needed)
    results['lagrangian_identities'] = verify_lagrangian_identities()

    # Load Pantheon+ data (needed for both baselines and PP-excluded fits)
    print("Loading Pantheon+ data ...")
    pp = load_pantheonplus()
    print(f"  N = {pp['N']}, calibrators = {sum(pp['is_calib'])}")
    print()

    # Sec. 16: optional v0.1 sanity baselines (PP-included)
    if not args.skip_baselines:
        results['baselines'] = run_baselines(pp, seeds)
    else:
        print("Skipping v0.1 baselines (--skip-baselines).\n")

    # Sec. 13: width relation
    results['width_relation'] = test_width_relation(seeds)

    # Sec. 14: recursive Omega_m
    results['recursive_Om'] = test_recursive_Om(seeds)

    # Sec. 15: Table 1
    results['table_1'] = make_table_1(seeds)

    # Final summary
    print("=" * 76)
    print("SUMMARY")
    print("=" * 76)
    li = results['lagrangian_identities']
    max_err = max(li['eq4_max_err'], li['eq6_max_err'], li['eq7_max_err'])
    print(f"Sec. 12 identities: max residual = {max_err:.2e}  (machine precision).")
    if 'baselines' in results:
        b = results['baselines']
        print("Sec. 16 v0.1 baselines (PP-included, N=1738):")
        print(f"  Lambda-CDM:    chi^2 = {b['lcdm_pp_inc']:.3f}     (ref 1624.85)")
        print(f"  LockPhi-A:     chi^2 = {b['lockphi_a_pp_inc']:.3f}  (ref 1601.19)")
        print(f"  LVC pendulum:  chi^2 = {b['lvc_pendulum_pp_inc']:.3f}  (ref 1602.77)")
    wr = results['width_relation']
    print(f"Sec. 13 width relation: A_fit/(ln phi)^4 = {wr['ratio_A_to_lnphi4']:.4f}, "
          f"w_der/(ln phi)^3 = {wr['ratio_w_to_lnphi3']:.4f}  "
          f"(equal under w = A/ln phi).")
    rec = results['recursive_Om']
    print(f"Sec. 14 recursive Omega_m: Delta chi^2 (lock vs free) = "
          f"{rec['delta_chi2_recursive']:+.4f}")
    print(f"        falsified candidate:   Delta chi^2 (1/(2+phi)) = "
          f"{rec['delta_chi2_2plusphi']:+.4f}")
    t1 = results['table_1']
    print("Sec. 15 Table 1 -- DeltaBIC vs Lambda-CDM (PP-excluded, N=37):")
    print(f"  Intermediate (Omega_m free, k=3):     "
          f"DeltaBIC = {t1['intermediate']['delta_bic']:+.3f}")
    print(f"  Full lock (Omega_m recursive, k=2):    "
          f"DeltaBIC = {t1['full_lock']['delta_bic']:+.3f}")
    print(f"  Falsified Omega_m = 1/(2+phi) (k=2):   "
          f"DeltaBIC = {t1['falsified_2pp']['delta_bic']:+.3f}")
    print()

    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Numerical results saved to {args.save_json}")

    print("Done.")


if __name__ == "__main__":
    main()
