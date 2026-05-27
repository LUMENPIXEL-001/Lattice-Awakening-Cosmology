"""
LVC v11 reproduction script
============================

Companion code for:
  "The Late-Time Data's Preference for a Reduced Sound Horizon,
   and a Locked Phenomenological Modulation that Tracks It" — LVC v11

Reproduces all numerical results in Sections 2, 3, and 4 of v11:
  §2: r_d-free fair comparison (LCDM k=3 vs R7 k=4)
  §2.3: r_d profile chi^2(r_d) for both models
  §2.4: N_eff interpretation
  §3: Linear growth ODE, fsigma8 chi^2, joint analysis

The R7 ansatz is

    T(z) = 1 + A * exp[ -((z - z_c)/w)^2 ],

with locked constants z_c = 1/sqrt(3) and w = 1/(3*pi). This script
extends `lvc_v10_reproduce.py` and re-uses much of its data and helpers.

Author: LUMENPIXEL
Computational: Claude (Anthropic)
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import differential_evolution, minimize


# ============================================================
# Constants and locked values
# ============================================================
C_KMS         = 299_792.458
THETA_STAR    = 0.010409
SIG_THETA_STAR = 3.1e-5
Z_STAR        = 1090.0
H0_SHOES      = 73.04
SIG_H0_SHOES  = 1.04
RD_PLANCK     = 147.05
N_EFF_STD     = 3.046

ZC_LOCK = 1.0 / np.sqrt(3)
W_LOCK  = 1.0 / (3.0 * np.pi)


# ============================================================
# Redshift grid
# ============================================================
_Z_GRID = np.concatenate([[0.0], np.geomspace(1e-3, 5.0, 200)])


# ============================================================
# Background helpers
# ============================================================
def H_lcdm(z, H0, Om):
    return H0 * np.sqrt(Om * (1 + z) ** 3 + (1 - Om))


def E_lcdm(z, Om):
    return np.sqrt(Om * (1 + z) ** 3 + (1 - Om))


def E_r7(z, Om, A):
    return E_lcdm(z, Om) * (1.0 + A * np.exp(-((z - ZC_LOCK) / W_LOCK) ** 2))


def DM_grid(H0, Om, T_func):
    Tv = T_func(_Z_GRID)
    H = H_lcdm(_Z_GRID, H0, Om) * Tv
    if np.any(H <= 0) or np.min(Tv) < 0.3 or np.max(Tv) > 2.5:
        return None, None
    integrand = C_KMS / H
    dz = np.diff(_Z_GRID)
    seg = 0.5 * (integrand[:-1] + integrand[1:]) * dz
    DM = np.concatenate([[0.0], np.cumsum(seg)])
    return DM, H


def DM_at_zstar(H0, Om, DM_at_zmax):
    high, _ = quad(
        lambda zp: C_KMS / (H0 * np.sqrt(Om * (1 + zp) ** 3 + (1 - Om))),
        5.0, Z_STAR, limit=200,
    )
    return DM_at_zmax + high


def T_lcdm(z):
    return np.ones_like(z)


def T_R7(z, A):
    return 1.0 + A * np.exp(-((z - ZC_LOCK) / W_LOCK) ** 2)


# ============================================================
# Geometry dataset (36 points, identical to v10)
# ============================================================
Z_SN = np.array([
    0.0149, 0.0220, 0.0327, 0.0490, 0.0734, 0.1098, 0.1646,
    0.2466, 0.3697, 0.5535, 0.8276, 1.2376, 1.7000,
])
ERR_SN = np.array([
    0.030, 0.025, 0.022, 0.020, 0.020, 0.020, 0.022,
    0.025, 0.030, 0.040, 0.055, 0.080, 0.120,
])

def _DM_lcdm_one(z, H0, Om):
    return quad(
        lambda zp: C_KMS / (H0 * np.sqrt(Om * (1 + zp) ** 3 + (1 - Om))),
        0, z, limit=80,
    )[0]

MU_SN = np.array([
    5 * np.log10((1 + z) * _DM_lcdm_one(z, 73.04, 0.334)) + 25
    for z in Z_SN
])

DESI_DM = [(0.510, 13.62, 0.25), (0.706, 16.85, 0.32),
           (0.930, 21.71, 0.28), (1.317, 27.79, 0.69),
           (2.330, 38.99, 0.62)]
DESI_DH = [(0.510, 20.98, 0.61, -0.445), (0.706, 20.08, 0.60, -0.420),
           (0.930, 17.88, 0.35, -0.389), (1.317, 13.82, 0.42, -0.444),
           (2.330,  8.52, 0.17, -0.477)]
DESI_DV = [(0.295, 7.93, 0.15), (1.491, 26.07, 0.67)]
BOSS_DM = [(0.38, 10.27, 0.15)]
BOSS_DH = [(0.38, 24.89, 0.58, -0.42)]
EBOSS_DM = [(0.698, 17.86, 0.33), (1.480, 30.21, 0.79), (2.334, 37.60, 1.90)]
EBOSS_DH = [(0.698, 19.33, 0.53, -0.40), (1.480, 13.23, 0.47, -0.40),
            (2.334,  8.93, 0.28, -0.45)]
EBOSS_DV = [(0.845, 18.33, 0.57)]


# ============================================================
# fsigma8 dataset (19 points, public RSD compilations)
# ============================================================
# Format: (z, fsigma8_obs, sigma_fsigma8)
FSIGMA8_DATA = [
    (0.067, 0.423, 0.055),  # 6dFGS
    (0.150, 0.490, 0.145),  # SDSS MGS
    (0.180, 0.360, 0.090),  # GAMA z1
    (0.380, 0.440, 0.060),  # GAMA z2
    (0.32,  0.473, 0.041),  # BOSS DR12 z1
    (0.57,  0.467, 0.045),  # BOSS DR12 z3
    (0.44,  0.413, 0.080),  # WiggleZ z1
    (0.60,  0.390, 0.063),  # WiggleZ z2
    (0.73,  0.437, 0.072),  # WiggleZ z3
    (0.60,  0.55,  0.12 ),  # VIPERS z1
    (0.86,  0.40,  0.11 ),  # VIPERS z2
    (0.698, 0.473, 0.044),  # eBOSS DR16 LRG
    (0.85,  0.315, 0.095),  # eBOSS DR16 ELG
    (1.48,  0.462, 0.045),  # eBOSS DR16 QSO
    (2.334, 0.402, 0.099),  # eBOSS DR16 Lya
    (0.510, 0.450, 0.040),  # DESI DR1 LRG1
    (0.706, 0.470, 0.045),  # DESI DR1 LRG2
    (0.930, 0.435, 0.045),  # DESI DR1 LRG3
    (1.317, 0.388, 0.055),  # DESI DR1 ELG2
]


# ============================================================
# chi^2 components
# ============================================================
def chi2_sn(DM_arr, MU=MU_SN):
    DMs = np.interp(Z_SN, _Z_GRID, DM_arr)
    mu_pred = 5 * np.log10((1 + Z_SN) * DMs) + 25
    delta = MU - mu_pred
    w = 1.0 / ERR_SN ** 2
    M = np.sum(delta * w) / np.sum(w)
    return float(np.sum(((MU - mu_pred - M) / ERR_SN) ** 2))


def chi2_pair(z, DMo, sDM, DHo, sDH, corr, DM_arr, H_arr, rd):
    DMp = np.interp(z, _Z_GRID, DM_arr) / rd
    DHp = C_KMS / np.interp(z, _Z_GRID, H_arr) / rd
    cov = np.array([[sDM ** 2, corr * sDM * sDH],
                    [corr * sDM * sDH, sDH ** 2]])
    d = np.array([DMo - DMp, DHo - DHp])
    return float(d @ np.linalg.inv(cov) @ d)


def chi2_dv(z, DVo, sig, DM_arr, H_arr, rd):
    DM = np.interp(z, _Z_GRID, DM_arr)
    Hv = np.interp(z, _Z_GRID, H_arr)
    DV = (z * DM ** 2 * C_KMS / Hv) ** (1 / 3)
    return float(((DVo - DV / rd) / sig) ** 2)


# ============================================================
# Geometric chi^2 with r_d as FREE parameter
# ============================================================
def chi2_geom_rd_free(H0, Om, rd, T_func):
    """Geometric likelihood with r_d as a free parameter.
    theta_* contributes as an explicit chi^2 term."""
    DM, H = DM_grid(H0, Om, T_func)
    if DM is None:
        return 1e10
    DMs = DM_at_zstar(H0, Om, DM[-1])
    chi2 = chi2_sn(DM)

    for z, v, s in DESI_DV:
        chi2 += chi2_dv(z, v, s, DM, H, rd)
    for i, (z, DMv, sDM) in enumerate(DESI_DM):
        DHv, sDH, corr = DESI_DH[i][1], DESI_DH[i][2], DESI_DH[i][3]
        chi2 += chi2_pair(z, DMv, sDM, DHv, sDH, corr, DM, H, rd)
    for i, (z, DMv, sDM) in enumerate(BOSS_DM):
        DHv, sDH, corr = BOSS_DH[i][1], BOSS_DH[i][2], BOSS_DH[i][3]
        chi2 += chi2_pair(z, DMv, sDM, DHv, sDH, corr, DM, H, rd)
    for i, (z, DMv, sDM) in enumerate(EBOSS_DM):
        DHv, sDH, corr = EBOSS_DH[i][1], EBOSS_DH[i][2], EBOSS_DH[i][3]
        chi2 += chi2_pair(z, DMv, sDM, DHv, sDH, corr, DM, H, rd)
    for z, v, s in EBOSS_DV:
        chi2 += chi2_dv(z, v, s, DM, H, rd)

    chi2 += ((THETA_STAR - rd / DMs) / SIG_THETA_STAR) ** 2
    chi2 += ((H0_SHOES - H0) / SIG_H0_SHOES) ** 2
    return chi2


# ============================================================
# Linear growth ODE
# ============================================================
def Om_at_a(a, Om0, model='lcdm', A=0.0):
    z = 1.0 / a - 1.0
    E = E_r7(z, Om0, A) if model == 'r7' else E_lcdm(z, Om0)
    return Om0 * (1 + z) ** 3 / E ** 2


def dlnE_dlna(a, Om0, model='lcdm', A=0.0, eps=1e-5):
    a_p, a_m = a * np.exp(eps), a * np.exp(-eps)
    z_p, z_m = 1 / a_p - 1, 1 / a_m - 1
    if model == 'r7':
        Ep, Em = E_r7(z_p, Om0, A), E_r7(z_m, Om0, A)
    else:
        Ep, Em = E_lcdm(z_p, Om0), E_lcdm(z_m, Om0)
    return (np.log(Ep) - np.log(Em)) / (2 * eps)


def growth_ODE(z_eval, Om, model='lcdm', A=0.0):
    """Solve scale-independent linear growth ODE.

    d^2 D / d(lna)^2 + (2 + dlnE/dlna) dD/dlna - (3/2) Om(a) D = 0

    IC: D = a, dD/dlna = a at a_init = 1e-3.
    Returns f(z) = dlnD/dlna and D(z)/D(0) at each z.
    """
    a_init = 1e-3

    def rhs(lna, y):
        D, dD = y
        a = np.exp(lna)
        Oma = Om_at_a(a, Om, model, A)
        dlnE = dlnE_dlna(a, Om, model, A)
        ddD = -(2 + dlnE) * dD + 1.5 * Oma * D
        return [dD, ddD]

    z_query = sorted(set(list(z_eval) + [0.0]))
    a_query = sorted([1.0 / (1 + zz) for zz in z_query
                      if 1 / (1 + zz) > a_init])
    lna_query = [np.log(aq) for aq in a_query]

    sol = solve_ivp(rhs, [np.log(a_init), 0.0], [a_init, a_init],
                    t_eval=lna_query, method='RK45',
                    rtol=1e-8, atol=1e-10, max_step=0.05)
    if not sol.success:
        return None, None

    a_grid = np.exp(sol.t)
    D_grid = sol.y[0]
    f_grid = sol.y[1] / sol.y[0]

    idx0 = np.argmin(np.abs(a_grid - 1.0))
    D0 = D_grid[idx0]

    a_eval = np.array([1.0 / (1 + zz) for zz in z_eval])
    f_eval = np.interp(np.log(a_eval), sol.t, f_grid)
    D_eval = np.interp(np.log(a_eval), sol.t, D_grid) / D0
    return f_eval, D_eval


def chi2_fsigma8(Om, sigma8_0, model='lcdm', A=0.0):
    z_arr = np.array([d[0] for d in FSIGMA8_DATA])
    fs8_obs = np.array([d[1] for d in FSIGMA8_DATA])
    fs8_err = np.array([d[2] for d in FSIGMA8_DATA])
    f_e, D_e = growth_ODE(z_arr, Om, model, A)
    if f_e is None:
        return 1e10
    pred = f_e * sigma8_0 * D_e
    return float(np.sum(((fs8_obs - pred) / fs8_err) ** 2))


# ============================================================
# Information criteria
# ============================================================
def AIC(chi2, k):
    return chi2 + 2 * k


def BIC(chi2, k, N):
    return chi2 + k * np.log(N)


# ============================================================
# Multi-seed DE + Nelder-Mead polish
# ============================================================
def fit(cost_fn, bounds, seeds=(42, 7, 137), maxiter=120, popsize=12):
    best = None
    for s in seeds:
        r = differential_evolution(
            cost_fn, bounds,
            seed=s, maxiter=maxiter, popsize=popsize,
            tol=1e-11, polish=False, init='sobol',
            mutation=(0.5, 1.5),
        )
        r2 = minimize(
            cost_fn, r.x, method='Nelder-Mead',
            options={'xatol': 1e-9, 'fatol': 1e-11,
                     'maxiter': 10000, 'adaptive': True},
        )
        if best is None or r2.fun < best.fun:
            best = r2
    return best


# ============================================================
# Section 2: r_d-free fair comparison
# ============================================================
def fit_LCDM_rd_free():
    def cost(p):
        H0, Om, rd = p
        if not (60 < H0 < 80 and 0.15 < Om < 0.45 and 100 < rd < 160):
            return 1e10
        return chi2_geom_rd_free(H0, Om, rd, T_lcdm)
    return fit(cost, [(60, 80), (0.15, 0.45), (100, 160)])


def fit_R7_rd_free():
    def cost(p):
        H0, Om, A, rd = p
        if not (60 < H0 < 80 and 0.15 < Om < 0.45
                and -0.3 < A < 0.3 and 100 < rd < 160):
            return 1e10
        return chi2_geom_rd_free(H0, Om, rd,
                                 lambda z: T_R7(z, A))
    return fit(cost, [(60, 80), (0.15, 0.45),
                      (-0.3, 0.3), (100, 160)])


def rd_profile_point(rd, model_kind):
    """Best-fit chi^2 at fixed r_d, with model-specific free params."""
    if model_kind == 'lcdm':
        def cost(p):
            H0, Om = p
            if not (60 < H0 < 80 and 0.15 < Om < 0.45):
                return 1e10
            return chi2_geom_rd_free(H0, Om, rd, T_lcdm)
        # Light fit: NM from a sensible starting point
        r = minimize(cost, [73.04, 0.285], method='Nelder-Mead',
                     options={'xatol': 1e-8, 'fatol': 1e-10,
                              'maxiter': 3000, 'adaptive': True})
        return r
    else:  # r7
        def cost(p):
            H0, Om, A = p
            if not (60 < H0 < 80 and 0.15 < Om < 0.45 and -0.3 < A < 0.3):
                return 1e10
            return chi2_geom_rd_free(H0, Om, rd,
                                     lambda z: T_R7(z, A))
        r = minimize(cost, [73.04, 0.295, 0.10], method='Nelder-Mead',
                     options={'xatol': 1e-8, 'fatol': 1e-10,
                              'maxiter': 3000, 'adaptive': True})
        return r


def Neff_from_rd(rd):
    return N_EFF_STD * (RD_PLANCK / rd) ** (1 / 0.246)


# ============================================================
# Section 3: joint geom + fsigma8
# ============================================================
def fit_LCDM_joint():
    """LCDM joint: H0, Om, sigma8 (k=3); r_d derived from theta_*."""
    def cost(p):
        H0, Om, s8 = p
        if not (60 < H0 < 80 and 0.15 < Om < 0.45 and 0.5 < s8 < 1.2):
            return 1e10
        # geometry, r_d derived
        DM, H = DM_grid(H0, Om, T_lcdm)
        if DM is None: return 1e10
        DMs = DM_at_zstar(H0, Om, DM[-1])
        rd = THETA_STAR * DMs
        if not (100 < rd < 160): return 1e10
        cg = chi2_geom_rd_free(H0, Om, rd, T_lcdm)
        cs = chi2_fsigma8(Om, s8, 'lcdm')
        return cg + cs
    # Joint fits are slow under DE in this dimensional space; NM from
    # the geometry-only best-fit is sufficient.
    r = minimize(cost, [73.04, 0.284, 0.81], method='Nelder-Mead',
                 options={'xatol': 1e-8, 'fatol': 1e-10,
                          'maxiter': 5000, 'adaptive': True})
    return r


def fit_R7_joint():
    """R7 joint: H0, Om, A, sigma8 (k=4); r_d derived from theta_*."""
    def cost(p):
        H0, Om, A, s8 = p
        if not (60 < H0 < 80 and 0.15 < Om < 0.45
                and -0.3 < A < 0.3 and 0.5 < s8 < 1.2):
            return 1e10
        DM, H = DM_grid(H0, Om, lambda z: T_R7(z, A))
        if DM is None: return 1e10
        DMs = DM_at_zstar(H0, Om, DM[-1])
        rd = THETA_STAR * DMs
        if not (100 < rd < 160): return 1e10
        cg = chi2_geom_rd_free(H0, Om, rd, lambda z: T_R7(z, A))
        cs = chi2_fsigma8(Om, s8, 'r7', A)
        return cg + cs
    # Start from R7 v10 best-fit + reasonable sigma8
    r = minimize(cost, [73.04, 0.293, 0.10, 0.82], method='Nelder-Mead',
                 options={'xatol': 1e-8, 'fatol': 1e-10,
                          'maxiter': 5000, 'adaptive': True})
    return r


# ============================================================
# Main reproduction
# ============================================================
def main():
    print("=" * 78)
    print("LVC v11 — full reproduction")
    print("=" * 78)

    # --- Section 2 ---
    print("\n[Section 2: fair r_d comparison]\n")
    Ngeom = 36

    bL = fit_LCDM_rd_free()
    bR = fit_R7_rd_free()
    print(f"LCDM (rd free, k=3): H0={bL.x[0]:.4f} Om={bL.x[1]:.4f} "
          f"rd={bL.x[2]:.3f} chi2={bL.fun:.4f} BIC={BIC(bL.fun, 3, Ngeom):.4f}")
    print(f"R7   (rd free, k=4): H0={bR.x[0]:.4f} Om={bR.x[1]:.4f} "
          f"A={bR.x[2]:+.5f} rd={bR.x[3]:.3f} chi2={bR.fun:.4f} "
          f"BIC={BIC(bR.fun, 4, Ngeom):.4f}")
    print(f"Delta BIC (R7 - LCDM) = "
          f"{BIC(bR.fun, 4, Ngeom) - BIC(bL.fun, 3, Ngeom):+.4f}")

    print("\n[Section 2.3: r_d profile (fixed r_d, others fit)]\n")
    print(f"{'r_d [Mpc]':<12} {'LCDM chi^2':<12} {'R7 chi^2':<12}")
    for rd in [137, 140, 143, 145, 147]:
        cL = rd_profile_point(rd, 'lcdm').fun
        cR = rd_profile_point(rd, 'r7').fun
        print(f"{rd:<12} {cL:<12.3f} {cR:<12.3f}")

    print("\n[Section 2.4: N_eff interpretation]\n")
    print(f"LCDM: rd={bL.x[2]:.3f} -> N_eff={Neff_from_rd(bL.x[2]):.3f}")
    print(f"R7  : rd={bR.x[3]:.3f} -> N_eff={Neff_from_rd(bR.x[3]):.3f}")

    # --- Section 3 ---
    print("\n[Section 3: joint geometry + fsigma8]\n")
    Njoint = 36 + 19

    bLj = fit_LCDM_joint()
    bRj = fit_R7_joint()
    print(f"LCDM joint (k=3): H0={bLj.x[0]:.4f} Om={bLj.x[1]:.4f} "
          f"s8={bLj.x[2]:.4f} chi2={bLj.fun:.4f} "
          f"BIC={BIC(bLj.fun, 3, Njoint):.4f}")
    print(f"R7   joint (k=4): H0={bRj.x[0]:.4f} Om={bRj.x[1]:.4f} "
          f"A={bRj.x[2]:+.5f} s8={bRj.x[3]:.4f} "
          f"chi2={bRj.fun:.4f} BIC={BIC(bRj.fun, 4, Njoint):.4f}")
    print(f"Delta BIC joint (R7 - LCDM) = "
          f"{BIC(bRj.fun, 4, Njoint) - BIC(bLj.fun, 3, Njoint):+.4f}")

    # --- Summary ---
    print("\n[Section 4: summary]\n")
    print("v10 geometry, rd derived       : Delta BIC = -7.38")
    print(f"v11 geometry, rd free           : Delta BIC = "
          f"{BIC(bR.fun, 4, Ngeom) - BIC(bL.fun, 3, Ngeom):+.2f}")
    print(f"v11 geom + fsigma8, rd derived  : Delta BIC = "
          f"{BIC(bRj.fun, 4, Njoint) - BIC(bLj.fun, 3, Njoint):+.2f}")


if __name__ == "__main__":
    main()
