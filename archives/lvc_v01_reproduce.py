"""
LVC v0.1 -- Single-file reproduction script
============================================

Reproduces all numerical results in the working paper:

    "Lagrangian Variable Cosmology: Theoretical Reconstruction of the
     Lock-phi Modulation under a Phase-Rotational Spacetime-Unfolding
     Hypothesis"
                                LUMENPIXEL, May 2026

This is a single self-contained Python file. No LVC-specific dependencies.
All constants, BAO data tables, distance-prior calibration, Pantheon+
loader, and fitter are inlined from v15/v16 conventions.


Working hypothesis
------------------
The construction proceeds from a working hypothesis that spacetime
unfolds through the rotational expansion of a compact phase variable,
whose dynamics on the logarithmic time coordinate theta = ln(1+z) take
the form of a parametric pendulum equation. This hypothesis is not
derived from first principles in the present paper; it is stated as the
starting point of the construction and is to be addressed in future
work.


The Friedmann equation
----------------------
The construction writes the Hubble rate as
    H^2(z) = H_0^2 * [ Om*(1+z)^3 + Omega_LVC(z) ]
with no separate cosmological-constant term. The dimensionless dark-energy
density Omega_LVC(z) is derived from the modulation factor T(z):
    Omega_LVC(z) = [ Om*(1+z)^3 + (1-Om) ] * T(z)^2  -  Om*(1+z)^3
The A=0 limit recovers the constant value (1 - Om), reproducing Lambda-CDM.

For nonzero A the function Omega_LVC(z) is not constant: it deviates by up
to ~21% near the critical point z_c = 1/phi and returns to a Lambda-like
value at high z. The corresponding effective equation of state crosses
w = -1 at the boundaries of the bound region.


Reproduces (combined likelihood, N=1738):
-----------------------------------------
    LVC pendulum (theta-Gaussian)        : chi^2 = 1602.77   (k=4)
    Lock-phi-A v16 (z-Gaussian)          : chi^2 = 1601.19   (k=4)
    A=0 special case (= Lambda-CDM)      : chi^2 = 1624.85   (k=3)

    Delta-BIC between A!=0 fit and A=0 special case:
        LVC pendulum            :  -14.62
        Lock-phi-A v16          :  -16.20
    Delta-chi^2 (LVC - LockPhi-A): +1.58


External data
-------------
The Pantheon+ data (~33 MB) and covariance must be available locally:
    Pantheon+SH0ES.dat
    Pantheon+SH0ES_STAT+SYS.cov

If absent, the script offers automatic git-based download from
github.com/PantheonPlusSH0ES/DataRelease (sub-directory
Pantheon+_Data/4_DISTANCES_AND_COVAR).

Set PANTHEONPLUS_DIR environment variable, or place files in ./PP_data/

Approximate runtime
-------------------
Default (single seed, fast):                 ~2 min
With --multiseed (5 seeds, paper precision): ~10 min
With --verify-ode (numerical ODE check):     +30 sec

Usage
-----
    python lvc_v01_reproduce.py                    # default fast run
    python lvc_v01_reproduce.py --multiseed        # paper-precision
    python lvc_v01_reproduce.py --verify-ode       # numerical ODE check
    python lvc_v01_reproduce.py --skip-baselines   # only LVC fit
    python lvc_v01_reproduce.py --help             # full options


Author          : LUMENPIXEL  (independent researcher, Busan, Korea)
Comp. assist.   : Claude (Anthropic)
Date            : May 2026
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
from scipy.optimize import differential_evolution, minimize


# ============================================================================
# SECTION 1. CONSTANTS
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
INV_PHI = 1.0 / PHI                       # 0.61803...

# Lock-phi-A parameters (v16 phenomenology, also used by LVC)
W_PHI_A = E / (5 * PI)                    # 0.17305

# LVC pendulum locked parameters (Axioms L3, L4)
THETA_C_LVC = np.log(PHI)                 # 0.48121  (= ln phi)
W_LVC       = E / (5 * PI)                # 0.17305  (= e / 5pi)
OMEGA_0_SQ  = 2.0 / W_LVC**2              # 50 pi^2 / e^2 = 66.7853
OMEGA_0     = np.sqrt(OMEGA_0_SQ)         # sqrt(2) * 5pi / e = 8.1722

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

# Combined-likelihood total measurement count (=1738)
N_COMBINED = 1701 + 13 + 11 + 2 + 7 + 3 + 1


# ============================================================================
# SECTION 2. PANTHEON+ DATA LOADER (with auto-download)
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
# SECTION 3. BACKGROUND COSMOLOGY
# ============================================================================

# Dense z-grid for line-of-sight integrals
_Z_GRID = np.concatenate([[0.0], np.geomspace(1e-3, 5.0, 200)])


def E_lcdm(z, Om):
    """Flat Lambda-CDM normalised Hubble rate."""
    return np.sqrt(Om*(1+z)**3 + (1-Om))


# ----------------------------------------------------------------------------
# Modulation factors
# ----------------------------------------------------------------------------
def T_zGauss(z, A, zc=INV_PHI, w=W_PHI_A):
    """v16 Lock-phi-A modulation: Gaussian on z."""
    return 1.0 + A * np.exp(-((z - zc)/w)**2)


def T_thetaGauss(z, A, theta_c=THETA_C_LVC, w=W_LVC):
    """LVC pendulum modulation: Gaussian on theta = ln(1+z).

    This is the closed-form solution of the LVC pendulum ODE
        v''(theta) + Omega^2(theta) * v = 0
    with Omega^2(theta) = (2/w^2) * [1 - 2*((theta-theta_c)/w)**2]
    and initial conditions v(theta_c) = A, v'(theta_c) = 0.
    """
    theta = np.log(1.0 + z)
    return 1.0 + A * np.exp(-((theta - theta_c)/w)**2)


# ----------------------------------------------------------------------------
# LVC Hubble rate: Friedmann equation with derived dark-energy density
# ----------------------------------------------------------------------------
def Omega_LVC(z, A, Om, theta_c=THETA_C_LVC, w=W_LVC):
    """Dimensionless dark-energy density of the LVC construction.

    The Friedmann equation of the construction is written as
        H^2(z) / H_0^2 = Om*(1+z)^3 + Omega_LVC(z)
    with no separate cosmological-constant term. Omega_LVC is derived from
    the modulation factor and recovers (1 - Om) in the A=0 limit.

    Definition:
        Omega_LVC(z) = [Om*(1+z)^3 + (1-Om)] * T(z)^2 - Om*(1+z)^3
    """
    T = T_thetaGauss(z, A, theta_c, w)
    return (Om*(1+z)**3 + (1-Om)) * T**2 - Om*(1+z)**3


def E_LVC_squared(z, A, Om, theta_c=THETA_C_LVC, w=W_LVC):
    """Normalised Hubble rate squared for the LVC construction:
       (H/H_0)^2 = Om*(1+z)^3 + Omega_LVC(z)
    Equivalent to E_lcdm(z)^2 * T(z)^2 by construction."""
    return Om*(1+z)**3 + Omega_LVC(z, A, Om, theta_c, w)


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
# SECTION 4. PANTHEON+ LIKELIHOOD
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
# SECTION 5. BAO DATA TABLES (DR1, DR2, BOSS, eBOSS)
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

# DESI DR1 BAO
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
# SECTION 6. DISTANCE PRIORS (sound-horizon multiplicative calibration)
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
# SECTION 7. COMBINED BAO LIKELIHOOD
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

    # ---- DR2 BAO ----
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

    # ---- DR1 BAO ----
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

    # ---- BOSS DR12 ----
    for (z, DMo, sDM), (z2, DHo, sDH, corr) in zip(BOSS_DM, BOSS_DH):
        DMp = np.interp(z, _Z_GRID, DC) / rd
        DHp = C_KMS / np.interp(z, _Z_GRID, H) / rd
        cov = np.array([[sDM**2, corr * sDM * sDH],
                        [corr * sDM * sDH, sDH**2]])
        d = np.array([DMo - DMp, DHo - DHp])
        chi2 += d @ np.linalg.inv(cov) @ d

    # ---- eBOSS DR16 ----
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


def chi2_total_combined(H0, Om, Ob_h2, T_func, pp):
    """PP + (DR1+DR2)BAO + SH0ES + DP."""
    DM_star = DM_at_zstar(H0, Om, Ob_h2, T_func)
    rd = THETA_STAR * DM_star
    chi_pp, _ = chi2_panplus(H0, Om, T_func, pp)
    chi_bao = chi2_bao_combined(H0, Om, T_func, rd)
    chi_h = ((H0_SHOES - H0) / SIG_H0_SHOES) ** 2
    chi_dp, _, _ = chi2_DP(H0, Om, Ob_h2, T_func)
    return chi_pp + chi_bao + chi_h + chi_dp


# ============================================================================
# SECTION 8. FITTERS
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
    """Fit Lambda-CDM (k=3): H0, Om, Ob_h2."""
    bounds = [(60, 80), (0.20, 0.45), (0.020, 0.025)]
    def cost(p):
        return chi2_total_combined(p[0], p[1], p[2], None, pp)
    return _de_nm(cost, bounds, seeds=seeds)


def fit_LockPhiA(pp, seeds):
    """Fit Lock-phi-A v16 (k=4): H0, Om, Ob_h2, A. z-Gaussian, locked zc, w."""
    bounds = [(60, 80), (0.20, 0.45), (0.020, 0.025), (-0.3, 0.3)]
    def cost(p):
        T = lambda z: T_zGauss(z, p[3])
        return chi2_total_combined(p[0], p[1], p[2], T, pp)
    return _de_nm(cost, bounds, seeds=seeds)


def fit_LVC(pp, seeds):
    """Fit LVC pendulum (k=4): H0, Om, Ob_h2, A. theta-Gaussian, locked theta_c, w."""
    bounds = [(60, 80), (0.20, 0.45), (0.020, 0.025), (-0.3, 0.3)]
    def cost(p):
        T = lambda z: T_thetaGauss(z, p[3])
        return chi2_total_combined(p[0], p[1], p[2], T, pp)
    return _de_nm(cost, bounds, seeds=seeds)


# ============================================================================
# SECTION 9. NUMERICAL ODE VERIFICATION (optional)
# ============================================================================
def verify_ode():
    """Numerical integration of the LVC pendulum ODE.

    ODE: v''(theta) + Omega^2(theta) * v = 0
         Omega^2(theta) = (2/w^2) * [1 - 2*((theta-theta_c)/w)**2]
    IC : v(theta_c) = A, v'(theta_c) = 0

    Verifies that the analytic Gaussian closed-form
         v(theta) = A * exp(-((theta-theta_c)/w)**2)
    is reproduced by direct integration to machine precision.
    """
    print("\n" + "=" * 70)
    print("ODE verification: numerical integration vs analytic Gaussian")
    print("=" * 70)

    A_test = 0.05
    theta_c = THETA_C_LVC
    w = W_LVC

    def rhs(theta, y):
        v, dv = y
        Om2 = (2.0/w**2) * (1.0 - 2.0*((theta - theta_c)/w)**2)
        return [dv, -Om2 * v]

    # Forward and backward from theta_c
    sol_f = solve_ivp(rhs, [theta_c, theta_c + 3*w], [A_test, 0.0],
                      method='DOP853', rtol=1e-12, atol=1e-15, dense_output=True)
    sol_b = solve_ivp(rhs, [theta_c, theta_c - 3*w], [A_test, 0.0],
                      method='DOP853', rtol=1e-12, atol=1e-15, dense_output=True)

    print(f"  theta_c = ln(phi)    = {theta_c:.8f}")
    print(f"  w       = e/(5 pi)   = {w:.8f}")
    print(f"  A       (test value) = {A_test}")
    print()
    print(f"  {'theta':>10} {'v_ODE':>14} {'v_analytic':>14} {'|diff|':>10}")
    print("  " + "-" * 50)
    max_err = 0.0
    for theta in np.linspace(theta_c - 2.5*w, theta_c + 2.5*w, 11):
        if theta >= theta_c:
            v_ode = float(sol_f.sol(theta)[0])
        else:
            v_ode = float(sol_b.sol(theta)[0])
        v_an = A_test * np.exp(-((theta - theta_c)/w)**2)
        err = abs(v_ode - v_an)
        max_err = max(max_err, err)
        print(f"  {theta:>10.5f} {v_ode:>14.10f} {v_an:>14.10f} {err:>10.2e}")
    print()
    print(f"  Maximum residual: {max_err:.2e}")
    if max_err < 1e-10:
        print("  Numerical ODE integration matches analytic closed form to 1e-10.")
    else:
        print(f"  WARNING: residual {max_err:.2e} larger than expected.")


# ============================================================================
# SECTION 10. MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="LVC v0.1 reproduction script",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--multiseed', action='store_true',
                        help="use 5 seeds per fit (paper precision, slower)")
    parser.add_argument('--skip-baselines', action='store_true',
                        help="skip Lambda-CDM and Lock-phi-A baselines")
    parser.add_argument('--verify-ode', action='store_true',
                        help="also run numerical ODE check")
    parser.add_argument('--save-json', metavar='PATH',
                        help="save numerical results as JSON")
    args = parser.parse_args()

    seeds = (42, 7, 13, 99, 31) if args.multiseed else (42,)
    t0 = time.time()

    print("=" * 70)
    print("LVC v0.1 Reproduction")
    print("=" * 70)
    print(f"  Pantheon+ data dir : {PP_DIR}")
    print(f"  Seeds              : {seeds}")
    print(f"  N (combined)       : {N_COMBINED}")
    print()

    # Print locked LVC parameters
    print("LVC pendulum locked parameters (Axioms L3, L4):")
    print(f"  theta_c = ln(phi)    = {THETA_C_LVC:.6f}")
    print(f"  w       = e/(5 pi)   = {W_LVC:.6f}")
    print(f"  Omega_0 = sqrt(2)*5pi/e = {OMEGA_0:.6f}")
    print(f"          (Omega_0^2  = 2/w^2 = 50 pi^2/e^2 = {OMEGA_0_SQ:.6f})")
    print()

    if args.verify_ode:
        verify_ode()

    pp = load_pantheonplus()
    print(f"Pantheon+ loaded: N = {pp['N']}")
    print()

    results = {}
    chi_LCDM = chi_LP = chi_LVC = None
    x_LCDM   = x_LP   = x_LVC   = None

    if not args.skip_baselines:
        # ---- Lambda-CDM ----
        print("Fitting Lambda-CDM (k=3)...")
        ti = time.time()
        chi_LCDM, x_LCDM = fit_LCDM(pp, seeds)
        print(f"  chi^2 = {chi_LCDM:.4f}    (time: {time.time()-ti:.1f}s)")
        print(f"  H0 = {x_LCDM[0]:.4f}, Om = {x_LCDM[1]:.4f}, "
              f"Ob_h^2 = {x_LCDM[2]:.5f}")
        print()

        # ---- Lock-phi-A v16 ----
        print("Fitting Lock-phi-A v16 (z-Gaussian, k=4)...")
        ti = time.time()
        chi_LP, x_LP = fit_LockPhiA(pp, seeds)
        print(f"  chi^2 = {chi_LP:.4f}    (time: {time.time()-ti:.1f}s)")
        print(f"  H0 = {x_LP[0]:.4f}, Om = {x_LP[1]:.4f}, "
              f"Ob_h^2 = {x_LP[2]:.5f}, A = {x_LP[3]:+.5f}")
        print()

    # ---- LVC pendulum ----
    print("Fitting LVC pendulum (theta-Gaussian, k=4)...")
    ti = time.time()
    chi_LVC, x_LVC = fit_LVC(pp, seeds)
    print(f"  chi^2 = {chi_LVC:.4f}    (time: {time.time()-ti:.1f}s)")
    print(f"  H0 = {x_LVC[0]:.4f}, Om = {x_LVC[1]:.4f}, "
          f"Ob_h^2 = {x_LVC[2]:.5f}, A = {x_LVC[3]:+.5f}")
    print()

    # ---- Derived Omega_LVC(z) at best fit ----
    print("Derived dark-energy density Omega_LVC(z) at best fit:")
    print(f"  (Friedmann eq: H^2/H_0^2 = Om*(1+z)^3 + Omega_LVC(z))")
    print(f"  {'z':>6}  {'Omega_LVC':>10}  {'(1-Om)':>10}  {'dev_%':>8}")
    Om_fit = x_LVC[1]
    A_fit = x_LVC[3]
    for z_v in [0.0, INV_PHI, 0.83, 1.0, 2.0, 5.0]:
        OmLVC = Omega_LVC(z_v, A_fit, Om_fit)
        OmLambda = 1 - Om_fit
        dev = (OmLVC - OmLambda)/OmLambda * 100
        print(f"  {z_v:>6.4f}  {OmLVC:>10.5f}  {OmLambda:>10.5f}  {dev:>+8.2f}%")

    # Effective w(z)
    print()
    print("Effective equation of state w_eff(z) at best fit:")
    print(f"  (w_eff = -1 - (1/3)*d ln Omega_LVC / d ln(1+z))")
    print(f"  {'z':>6}  {'w_eff':>9}")
    dz = 1e-4
    for z_v in [0.0, 0.30, INV_PHI, 0.83, 1.0, 2.0]:
        r0 = Omega_LVC(z_v, A_fit, Om_fit)
        r1 = Omega_LVC(z_v + dz, A_fit, Om_fit)
        if r0 > 0 and r1 > 0:
            dlnr = (np.log(r1) - np.log(r0)) / dz
            dlnzp1 = 1.0/(1+z_v)
            w_e = -1 - (1.0/3.0) * (dlnr / dlnzp1)
            print(f"  {z_v:>6.4f}  {w_e:>+9.4f}")
    print()

    # ---- Summary table ----
    print("=" * 70)
    print(f"Summary ({'multi-seed' if args.multiseed else 'single-seed'}, "
          f"N = {N_COMBINED}):")
    print("=" * 70)
    print(f"  {'Model':<40} {'k':>3} {'chi^2':>11}  {'Delta-BIC':>11}")
    print("  " + "-" * 70)

    if chi_LCDM is not None:
        dBIC_LVC = (chi_LVC - chi_LCDM) + 1*np.log(N_COMBINED)
        print(f"  {'LVC pendulum (theta-Gauss)':<40} {4:>3} "
              f"{chi_LVC:>11.4f}  {dBIC_LVC:>+11.2f}")
        if chi_LP is not None:
            dBIC_LP  = (chi_LP  - chi_LCDM) + 1*np.log(N_COMBINED)
            print(f"  {'Lock-phi-A v16 (z-Gauss)':<40} {4:>3} "
                  f"{chi_LP:>11.4f}  {dBIC_LP:>+11.2f}")
        print(f"  {'A=0 special case (= Lambda-CDM)':<40} {3:>3} "
              f"{chi_LCDM:>11.4f}  {0.0:>+11.2f}")
        print()
        print("  (Delta-BIC values are taken between each A!=0 fit")
        print("   and the A=0 special case of the construction.)")
    else:
        print(f"  {'LVC pendulum (theta-Gauss)':<40} {4:>3} "
              f"{chi_LVC:>11.4f}  {'(no baselines)':>11}")

    # Comparison vs LockPhi-A
    if chi_LP is not None:
        print()
        print(f"  Delta-chi^2 (LVC - Lock-phi-A) = {chi_LVC - chi_LP:+.4f}")
        if chi_LVC > chi_LP:
            print(f"    Lock-phi-A is better by {chi_LVC - chi_LP:.2f} chi^2 units.")
        else:
            print(f"    LVC is better by {chi_LP - chi_LVC:.2f} chi^2 units.")

    print()
    print(f"Total wall time: {time.time()-t0:.1f}s")

    # ---- Save JSON if requested ----
    if args.save_json:
        results = dict(
            N=N_COMBINED,
            seeds=list(seeds),
            theta_c=THETA_C_LVC, w=W_LVC, Omega_0=OMEGA_0,
            LCDM=dict(chi2=chi_LCDM,
                      params=x_LCDM.tolist() if x_LCDM is not None else None),
            LockPhiA=dict(chi2=chi_LP,
                          params=x_LP.tolist() if x_LP is not None else None),
            LVC=dict(chi2=chi_LVC,
                     params=x_LVC.tolist() if x_LVC is not None else None),
        )
        with open(args.save_json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.save_json}")


# ============================================================================
if __name__ == '__main__':
    main()
