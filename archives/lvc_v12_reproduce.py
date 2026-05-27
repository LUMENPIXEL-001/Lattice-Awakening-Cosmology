"""
LVC v12 reproduction script
============================

Reproduces all numerical results in:
  "A Family of Locked Phenomenological Modulations in Late-Time
   Cosmological Data: v10 and DESI DR2 Prefer Different Members"

Sections covered:
  §2 The DR1 lock does not survive DR2 (DR2 free fit)
  §3 The v10 family (lock candidates table)
  §4.1 Geometry only (LCDM, n=0, n=1, free)
  §4.2 r_d-free fair comparison
  §4.3 Joint geom + fsigma8

Author: LUMENPIXEL
Computational: Claude (Anthropic)
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import differential_evolution, minimize
from scipy.linalg import inv


# ============================================================
# Constants and locks
# ============================================================
C_KMS         = 299_792.458
THETA_STAR    = 0.010409
SIG_THETA_STAR = 3.1e-5
Z_STAR        = 1090.0
H0_SHOES      = 73.04
SIG_H0_SHOES  = 1.04

ZC0 = 1.0 / np.sqrt(3)        # 1/√3
W0  = 1.0 / (3.0 * np.pi)     # 1/(3π)

# Family: z_c(n) = ZC0 + n W0,  w(n) = (n+1) W0
def family_lock(n):
    return ZC0 + n * W0, (n + 1) * W0

# v10 lock = n=0; v12 lock = n=1
ZC_v10, W_v10 = family_lock(0)
ZC_v12, W_v12 = family_lock(1)


# ============================================================
# Redshift grid and background
# ============================================================
_Z_GRID = np.concatenate([[0.0], np.geomspace(1e-3, 5.0, 200)])

def H_lcdm(z, H0, Om):
    return H0 * np.sqrt(Om*(1+z)**3 + (1-Om))

def E_lcdm(z, Om):
    return np.sqrt(Om*(1+z)**3 + (1-Om))

def E_with_T(z, Om, T_func):
    return E_lcdm(z, Om) * T_func(z)

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
    high, _ = quad(lambda zp: C_KMS / (H0 * np.sqrt(Om*(1+zp)**3 + (1-Om))),
                   5.0, Z_STAR, limit=200)
    return DM_at_zmax + high


# ============================================================
# T_R: Gaussian modulation with given lock
# ============================================================
def T_R(zc, w, A):
    def T(z):
        return 1 + A * np.exp(-((z - zc) / w)**2)
    return T

T_lcdm = lambda z: np.ones_like(z)


# ============================================================
# DR1 dataset (DESI 2024 / DR1) - 36 points
# ============================================================
DESI_DR1_DV = [(0.295, 7.93, 0.15), (1.491, 26.07, 0.67)]
DESI_DR1_pair = [
    (0.510, 13.62, 0.25, 20.98, 0.61, -0.445),
    (0.706, 16.85, 0.32, 20.08, 0.60, -0.420),
    (0.930, 21.71, 0.28, 17.88, 0.35, -0.389),
    (1.317, 27.79, 0.69, 13.82, 0.42, -0.444),
    (2.330, 38.99, 0.62,  8.52, 0.17, -0.477),
]


# ============================================================
# DR2 dataset (DESI 2025) - 37 points
# ============================================================
DESI_DR2_DV = [(0.295, 7.942, 0.075)]
DESI_DR2_pair = [
    (0.510, 13.587, 0.169, 21.863, 0.427, -0.475),
    (0.706, 17.347, 0.180, 19.458, 0.332, -0.423),
    (0.934, 21.574, 0.153, 17.641, 0.193, -0.425),
    (1.321, 27.605, 0.320, 14.178, 0.217, -0.437),
    (1.484, 30.519, 0.758, 12.816, 0.513, -0.489),
    (2.330, 38.988, 0.531,  8.632, 0.101, -0.431),
]


# ============================================================
# Common: BOSS DR12, eBOSS DR16
# ============================================================
BOSS_DM = [(0.38, 10.27, 0.15)]
BOSS_DH = [(0.38, 24.89, 0.58, -0.42)]
EBOSS_DM = [(0.698, 17.86, 0.33), (1.480, 30.21, 0.79), (2.334, 37.60, 1.90)]
EBOSS_DH = [(0.698, 19.33, 0.53, -0.40), (1.480, 13.23, 0.47, -0.40),
            (2.334,  8.93, 0.28, -0.45)]
EBOSS_DV = [(0.845, 18.33, 0.57)]


# ============================================================
# Pantheon+ binned (same in DR1, DR2)
# ============================================================
Z_SN = np.array([0.0149, 0.0220, 0.0327, 0.0490, 0.0734, 0.1098, 0.1646,
                 0.2466, 0.3697, 0.5535, 0.8276, 1.2376, 1.7000])
ERR_SN = np.array([0.030, 0.025, 0.022, 0.020, 0.020, 0.020, 0.022,
                   0.025, 0.030, 0.040, 0.055, 0.080, 0.120])
def _DM1(z, H0, Om):
    return quad(lambda zp: C_KMS/(H0*np.sqrt(Om*(1+zp)**3+(1-Om))), 0, z, limit=80)[0]
MU_SN = np.array([5*np.log10((1+z) * _DM1(z, 73.04, 0.334)) + 25 for z in Z_SN])


# ============================================================
# fsigma8 dataset (19 RSD points)
# ============================================================
FSIGMA8_DATA = [
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


# ============================================================
# Geometry chi^2
# ============================================================
def _chi2_sn(DM_arr):
    DMs = np.interp(Z_SN, _Z_GRID, DM_arr)
    mu_pred = 5 * np.log10((1+Z_SN) * DMs) + 25
    delta = MU_SN - mu_pred
    w = 1.0 / ERR_SN**2
    M = np.sum(delta * w) / np.sum(w)
    return float(np.sum(((MU_SN - mu_pred - M) / ERR_SN)**2))

def _chi2_pair(z, DMo, sDM, DHo, sDH, corr, DM_arr, H_arr, rd):
    DMp = np.interp(z, _Z_GRID, DM_arr) / rd
    DHp = C_KMS / np.interp(z, _Z_GRID, H_arr) / rd
    cov = np.array([[sDM**2, corr*sDM*sDH], [corr*sDM*sDH, sDH**2]])
    d = np.array([DMo - DMp, DHo - DHp])
    return float(d @ inv(cov) @ d)

def _chi2_dv(z, DVo, sig, DM_arr, H_arr, rd):
    DM = np.interp(z, _Z_GRID, DM_arr)
    Hv = np.interp(z, _Z_GRID, H_arr)
    DV = (z * DM**2 * C_KMS / Hv) ** (1/3)
    return float(((DVo - DV/rd) / sig)**2)


def chi2_geom(H0, Om, T_func, dataset='DR2', rd=None):
    """
    dataset: 'DR1' or 'DR2'
    rd: if None, derive from theta_*; if provided, use it (rd-free mode)
    """
    DM, H = DM_grid(H0, Om, T_func)
    if DM is None: return 1e10
    DMs = DM_at_zstar(H0, Om, DM[-1])
    if rd is None:
        rd = THETA_STAR * DMs
    if not (100 < rd < 160): return 1e10

    chi2 = _chi2_sn(DM)

    if dataset == 'DR1':
        DV_list, pair_list = DESI_DR1_DV, DESI_DR1_pair
    else:
        DV_list, pair_list = DESI_DR2_DV, DESI_DR2_pair

    for z, v, s in DV_list:
        chi2 += _chi2_dv(z, v, s, DM, H, rd)
    for entry in pair_list:
        chi2 += _chi2_pair(*entry, DM, H, rd)

    for i, (z, DMv, sDM) in enumerate(BOSS_DM):
        DHv, sDH, corr = BOSS_DH[i][1], BOSS_DH[i][2], BOSS_DH[i][3]
        chi2 += _chi2_pair(z, DMv, sDM, DHv, sDH, corr, DM, H, rd)
    for i, (z, DMv, sDM) in enumerate(EBOSS_DM):
        DHv, sDH, corr = EBOSS_DH[i][1], EBOSS_DH[i][2], EBOSS_DH[i][3]
        chi2 += _chi2_pair(z, DMv, sDM, DHv, sDH, corr, DM, H, rd)
    for z, v, s in EBOSS_DV:
        chi2 += _chi2_dv(z, v, s, DM, H, rd)

    chi2 += ((THETA_STAR - rd/DMs) / SIG_THETA_STAR)**2
    chi2 += ((H0_SHOES - H0) / SIG_H0_SHOES)**2
    return chi2


# ============================================================
# Linear growth + fsigma8
# ============================================================
def chi2_fsigma8(Om, sigma8_0, T_func):
    """Solve linear growth ODE in T(z)-modified background, compute fsigma8 chi^2."""
    a_init = 1e-3
    def E(z):
        return E_with_T(z, Om, T_func)
    def Om_at_a(a):
        z = 1/a - 1
        return Om*(1+z)**3 / E(z)**2
    def dlnE_dlna(a, eps=1e-5):
        a_p, a_m = a*np.exp(eps), a*np.exp(-eps)
        z_p, z_m = 1/a_p - 1, 1/a_m - 1
        return (np.log(E(z_p)) - np.log(E(z_m))) / (2*eps)
    def rhs(lna, y):
        D, dD = y
        a = np.exp(lna)
        return [dD, -(2 + dlnE_dlna(a))*dD + 1.5*Om_at_a(a)*D]
    z_arr = np.array([d[0] for d in FSIGMA8_DATA])
    fs8_obs = np.array([d[1] for d in FSIGMA8_DATA])
    fs8_err = np.array([d[2] for d in FSIGMA8_DATA])
    z_query = sorted(set(list(z_arr) + [0.0]))
    a_query = sorted([1.0/(1+zz) for zz in z_query if 1/(1+zz) > a_init])
    lna_query = [np.log(aq) for aq in a_query]
    sol = solve_ivp(rhs, [np.log(a_init), 0.0], [a_init, a_init],
                    t_eval=lna_query, method='RK45', rtol=1e-8, atol=1e-10, max_step=0.05)
    if not sol.success:
        return 1e10
    a_grid = np.exp(sol.t); D_grid = sol.y[0]; f_grid = sol.y[1]/sol.y[0]
    idx0 = np.argmin(np.abs(a_grid - 1.0))
    D0 = D_grid[idx0]
    a_eval = np.array([1.0/(1+zz) for zz in z_arr])
    f_eval = np.interp(np.log(a_eval), sol.t, f_grid)
    D_eval = np.interp(np.log(a_eval), sol.t, D_grid)/D0
    pred = f_eval * sigma8_0 * D_eval
    return float(np.sum(((fs8_obs - pred)/fs8_err)**2))


# ============================================================
# Information criteria
# ============================================================
def BIC(chi2, k, N): return chi2 + k * np.log(N)
def AIC(chi2, k):    return chi2 + 2 * k


# ============================================================
# Multi-seed DE + NM
# ============================================================
def de_nm_fit(cost_fn, bounds, seeds=(42, 7, 137), maxiter=120, popsize=12):
    best = None
    for s in seeds:
        r = differential_evolution(cost_fn, bounds, seed=s, maxiter=maxiter,
                                    popsize=popsize, tol=1e-11, polish=False,
                                    init='sobol', mutation=(0.5, 1.5))
        r2 = minimize(cost_fn, r.x, method='Nelder-Mead',
                      options={'xatol': 1e-9, 'fatol': 1e-11,
                               'maxiter': 10000, 'adaptive': True})
        if best is None or r2.fun < best.fun:
            best = r2
    return best


# ============================================================
# Section 2: DR1 lock breakdown / DR2 free fit
# ============================================================
def fit_LCDM_geom(dataset='DR2'):
    def cost(p):
        H0, Om = p
        if not (60 < H0 < 80 and 0.15 < Om < 0.45): return 1e10
        return chi2_geom(H0, Om, T_lcdm, dataset)
    return de_nm_fit(cost, [(60, 80), (0.15, 0.45)])

def fit_R_locked_geom(zc, w, dataset='DR2'):
    def cost(p):
        H0, Om, A = p
        if not (60 < H0 < 80 and 0.15 < Om < 0.45 and -0.3 < A < 0.3): return 1e10
        return chi2_geom(H0, Om, T_R(zc, w, A), dataset)
    return de_nm_fit(cost, [(60, 80), (0.15, 0.45), (-0.3, 0.3)])

def fit_R_free_geom(dataset='DR2'):
    def cost(p):
        H0, Om, A, zc, w = p
        if not (60 < H0 < 80 and 0.15 < Om < 0.45 and -0.3 < A < 0.3
                and 0.05 < zc < 3.0 and 0.03 < w < 1.0):
            return 1e10
        return chi2_geom(H0, Om, T_R(zc, w, A), dataset)
    return de_nm_fit(cost, [(60, 80), (0.15, 0.45), (-0.3, 0.3),
                            (0.05, 3.0), (0.03, 1.0)],
                     seeds=(42, 7, 137, 99), maxiter=200, popsize=18)


# ============================================================
# Section 4.2: r_d-free
# ============================================================
def fit_LCDM_rd_free(dataset='DR2'):
    def cost(p):
        H0, Om, rd = p
        if not (60 < H0 < 80 and 0.15 < Om < 0.45 and 100 < rd < 160): return 1e10
        return chi2_geom(H0, Om, T_lcdm, dataset, rd=rd)
    return de_nm_fit(cost, [(60, 80), (0.15, 0.45), (100, 160)])

def fit_R12_rd_free(dataset='DR2'):
    def cost(p):
        H0, Om, A, rd = p
        if not (60 < H0 < 80 and 0.15 < Om < 0.45 and -0.3 < A < 0.3
                and 100 < rd < 160): return 1e10
        return chi2_geom(H0, Om, T_R(ZC_v12, W_v12, A), dataset, rd=rd)
    return de_nm_fit(cost, [(60, 80), (0.15, 0.45), (-0.3, 0.3), (100, 160)])


# ============================================================
# Section 4.3: joint
# ============================================================
def fit_LCDM_joint(dataset='DR2'):
    def cost(p):
        H0, Om, s8 = p
        if not (60 < H0 < 80 and 0.15 < Om < 0.45 and 0.5 < s8 < 1.2): return 1e10
        cg = chi2_geom(H0, Om, T_lcdm, dataset)
        if cg > 1e9: return cg
        return cg + chi2_fsigma8(Om, s8, T_lcdm)
    return minimize(cost, [73.04, 0.282, 0.81], method='Nelder-Mead',
                    options={'xatol':1e-8,'fatol':1e-10,'maxiter':5000,'adaptive':True})

def fit_R12_joint(dataset='DR2'):
    def cost(p):
        H0, Om, A, s8 = p
        if not (60 < H0 < 80 and 0.15 < Om < 0.45 and -0.3 < A < 0.3
                and 0.5 < s8 < 1.2):
            return 1e10
        T = T_R(ZC_v12, W_v12, A)
        cg = chi2_geom(H0, Om, T, dataset)
        if cg > 1e9: return cg
        return cg + chi2_fsigma8(Om, s8, T)
    return minimize(cost, [73.04, 0.292, 0.05, 0.82], method='Nelder-Mead',
                    options={'xatol':1e-8,'fatol':1e-10,'maxiter':5000,'adaptive':True})


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 78)
    print("LVC v12 — full reproduction")
    print("=" * 78)
    print(f"\nv10 lock (n=0): zc = 1/√3 = {ZC_v10:.5f},  w = 1/(3π) = {W_v10:.5f}")
    print(f"v12 lock (n=1): zc = 1/√3 + 1/(3π) = {ZC_v12:.5f},  w = 2/(3π) = {W_v12:.5f}")

    print("\n[Section 2: DR2 free 5-parameter fit]\n")
    bF = fit_R_free_geom('DR2')
    print(f"DR2 free: H0={bF.x[0]:.4f}  Om={bF.x[1]:.4f}  A={bF.x[2]:+.5f}")
    print(f"          zc={bF.x[3]:.5f}  w={bF.x[4]:.5f}  chi2={bF.fun:.4f}")
    print(f"  vs v10 lock zc={ZC_v10:.5f}, w={W_v10:.5f} -> diff: {(bF.x[3]/ZC_v10-1)*100:+.2f}%, {(bF.x[4]/W_v10-1)*100:+.2f}%")
    print(f"  vs n=1 lock zc={ZC_v12:.5f}, w={W_v12:.5f} -> diff: {(bF.x[3]/ZC_v12-1)*100:+.2f}%, {(bF.x[4]/W_v12-1)*100:+.2f}%")

    print("\n[Section 3: family table - n=0,1,2,3 across DR1, DR2]\n")
    print(f"{'n':<3} {'zc':<10} {'w':<10} {'DR1 χ²':<10} {'DR2 χ²':<10}")
    for n in [0, 1, 2, 3]:
        zc, w = family_lock(n)
        b1 = fit_R_locked_geom(zc, w, 'DR1')
        b2 = fit_R_locked_geom(zc, w, 'DR2')
        print(f"{n:<3} {zc:<10.5f} {w:<10.5f} {b1.fun:<10.3f} {b2.fun:<10.3f}")

    print("\n[Section 4.1: geometry only on DR2]\n")
    bL = fit_LCDM_geom('DR2')
    bn0 = fit_R_locked_geom(*family_lock(0), 'DR2')
    bn1 = fit_R_locked_geom(*family_lock(1), 'DR2')
    N = 37
    print(f"LCDM      (k=2): chi2={bL.fun:.4f} BIC={BIC(bL.fun,2,N):.4f}")
    print(f"R7 (n=0)  (k=3): chi2={bn0.fun:.4f} BIC={BIC(bn0.fun,3,N):.4f}  ΔBIC={BIC(bn0.fun,3,N)-BIC(bL.fun,2,N):+.4f}")
    print(f"R12(n=1)  (k=3): chi2={bn1.fun:.4f} BIC={BIC(bn1.fun,3,N):.4f}  ΔBIC={BIC(bn1.fun,3,N)-BIC(bL.fun,2,N):+.4f}")
    print(f"R7 free   (k=5): chi2={bF.fun:.4f} BIC={BIC(bF.fun,5,N):.4f}  ΔBIC={BIC(bF.fun,5,N)-BIC(bL.fun,2,N):+.4f}")

    print("\n[Section 4.2: r_d-free fair comparison on DR2]\n")
    bLrd = fit_LCDM_rd_free('DR2')
    bRrd = fit_R12_rd_free('DR2')
    print(f"LCDM rd-free (k=3): H0={bLrd.x[0]:.4f} Om={bLrd.x[1]:.4f} rd={bLrd.x[2]:.3f}")
    print(f"  chi2={bLrd.fun:.4f} BIC={BIC(bLrd.fun,3,N):.4f}")
    print(f"R12 rd-free (k=4): H0={bRrd.x[0]:.4f} Om={bRrd.x[1]:.4f} A={bRrd.x[2]:+.5f} rd={bRrd.x[3]:.3f}")
    print(f"  chi2={bRrd.fun:.4f} BIC={BIC(bRrd.fun,4,N):.4f}")
    print(f"  ΔBIC = {BIC(bRrd.fun,4,N)-BIC(bLrd.fun,3,N):+.4f}")

    print("\n[Section 4.3: joint geom + fsigma8 on DR2]\n")
    bLj = fit_LCDM_joint('DR2')
    bRj = fit_R12_joint('DR2')
    Nj = 37 + 19
    print(f"LCDM joint (k=3): chi2={bLj.fun:.4f} BIC={BIC(bLj.fun,3,Nj):.4f}")
    print(f"  H0={bLj.x[0]:.4f} Om={bLj.x[1]:.4f} s8={bLj.x[2]:.4f}")
    print(f"R12 joint (k=4): chi2={bRj.fun:.4f} BIC={BIC(bRj.fun,4,Nj):.4f}")
    print(f"  H0={bRj.x[0]:.4f} Om={bRj.x[1]:.4f} A={bRj.x[2]:+.5f} s8={bRj.x[3]:.4f}")
    print(f"  ΔBIC = {BIC(bRj.fun,4,Nj)-BIC(bLj.fun,3,Nj):+.4f}")


if __name__ == "__main__":
    main()
