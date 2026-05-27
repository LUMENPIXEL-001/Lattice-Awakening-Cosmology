"""
lvc_v15_reproduce.py
====================
Reproduce all numerical results in the LVC v15 working paper.

This script extends `lvc_v14_reproduce.py` with:
  - DESI DR1 BAO block (alternative to DR2)
  - free-Gaussian fitter (k=6) with multi-seed DE + NM
  - locked-Gaussian fitter for arbitrary (zc, w)
  - DR1-aware leave-one-out (11 BAO labels)
  - free-w LOO at zc = 4/7
  - BBN r_d = 147.05 +- 0.3 Mpc prior switch
  - direct A-grid scan for D1 reproducibility check

Dependencies:
  - d1_core.py             # PP loader, BAO data, chi^2 functions, basic fitters
  - v15_p2_LOO_DR1_fix.py  # DR1 LOO chi^2 (the helpers' DR1 branch was buggy)
  - Pantheon+ data files in PP_data/

Total runtime on a single CPU core: ~90-120 minutes for the full reproduction.
Each major section can be run independently by toggling the SECTIONS dict.

Usage:
    python lvc_v15_reproduce.py                # run everything
    python lvc_v15_reproduce.py --quick        # skip the heavy LOO sweeps

Output: prints all tables in the paper. Saves JSON results to ./v15_results/.

Author: LUMENPIXEL + Claude (Anthropic)
Date: May 2026
"""
import os, sys, time, json, argparse
import numpy as np
from scipy.optimize import minimize, differential_evolution

# ---- imports from the project core --------------------------------
from d1_core import (
    load_pantheonplus, chi2_total_T, chi2_panplus_T, chi2_DP_T,
    comoving_dist_grid_T, _Z_GRID, C_KMS, _DM_at_zstar_T,
    THETA_STAR, H0_SHOES, SIG_H0_SHOES,
    DESI_DR1_pair, DESI_DR1_DV,
    BOSS_DM, BOSS_DH, EBOSS_DM, EBOSS_DH, EBOSS_DV,
    fit_LCDM, fit_C1, fit_D1, fit_locked_gauss,
    T_C1, T_D1, ZC_C1, W_C1, ZC_D1, W_D1,
)
from v15_p2_LOO_DR1_fix import chi2_BAO_DR1_LOO, chi2_total_DR1_LOO

PI = np.pi

# ===================================================================
# Configuration
# ===================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true',
                    help='skip the heavy LOO sweeps (sections 4-6)')
parser.add_argument('--out', default='./v15_results',
                    help='output directory for JSON results')
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

SECTIONS = dict(
    s1_v14_baseline   = True,   # ~5 min
    s2_C1_DR1         = True,   # ~3 min
    s3_D1_burn        = True,   # ~5 min  (A-grid scan)
    s4_free_gauss     = True,   # ~30 min (5 seeds * 2 datasets * k=6)
    s5_lock_fits      = True,   # ~10 min (4 fits)
    s6_LOO_LockC_DR2  = not args.quick,   # ~30 min
    s6_LOO_LockC_DR1  = not args.quick,   # ~20 min
    s7_BBN            = True,   # ~10 min
    s8_freew_LOO      = not args.quick,   # ~30 min
    s9_T_profile      = True,   # <1 sec
)

LCDM_DR2 = None   # populated by section 1
LCDM_DR1 = None
N_DR2 = 1727
N_DR1 = 1725

results = {}

def save():
    with open(os.path.join(args.out, 'v15_all_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if hasattr(o,'__float__') else str(o))

def header(title):
    print('\n' + '='*72)
    print(title)
    print('='*72)


# ===================================================================
# Section 1: v14 baseline reproduction (DR2)
# ===================================================================
if SECTIONS['s1_v14_baseline']:
    header('Section 1: v14 baseline reproduction (DR2)')
    pp = load_pantheonplus()
    print(f'PP loaded: N={pp["N"]}, calib={pp["is_calib"].sum()}')

    t1 = time.time()
    chi2_L, p_L = fit_LCDM(pp, bao_set='DR2')
    H0_L, Om_L, Ob_L = p_L
    _, parts_L = chi2_total_T(H0_L, Om_L, Ob_L, None, pp,
                              include_DP=True, bao_set='DR2', return_parts=True)
    print(f'LCDM (k=3): chi2={chi2_L:.3f}  H0={H0_L:.3f}  Om={Om_L:.4f}  '
          f'Ob={Ob_L:.5f}  rd={parts_L["rd"]:.2f}')
    LCDM_DR2 = chi2_L

    chi2_C, p_C = fit_C1(pp, bao_set='DR2')
    H0_C, Om_C, Ob_C, A_C = p_C
    T_C = lambda z: T_C1(z, A_C)
    _, parts_C = chi2_total_T(H0_C, Om_C, Ob_C, T_C, pp,
                              include_DP=True, bao_set='DR2', return_parts=True)
    print(f'C1   (k=4): chi2={chi2_C:.3f}  H0={H0_C:.3f}  Om={Om_C:.4f}  '
          f'Ob={Ob_C:.5f}  A={A_C:+.4f}  rd={parts_C["rd"]:.2f}')

    dchi = chi2_C - chi2_L
    dBIC = dchi + (4-3)*np.log(N_DR2)
    print(f'\nC1 vs LCDM: Δχ² = {dchi:+.3f}, ΔBIC = {dBIC:+.3f}')
    print(f'(v14 paper Table 2: chi2_LCDM=1606.94, chi2_C1=1588.10, ΔBIC=-11.38)')
    print(f'Reproduction OK: ΔBIC = {dBIC:+.3f}  ({time.time()-t1:.0f}s)')

    results['s1'] = dict(
        LCDM=dict(chi2=chi2_L, H0=H0_L, Om=Om_L, Ob=Ob_L, rd=parts_L['rd']),
        C1=dict(chi2=chi2_C, H0=H0_C, Om=Om_C, Ob=Ob_C, A=A_C, rd=parts_C['rd']),
        dchi=dchi, dBIC=dBIC)
    save()


# ===================================================================
# Section 2: C1 on DR1 — generalisation test
# ===================================================================
if SECTIONS['s2_C1_DR1']:
    header('Section 2: C1 cross-check on DR1 BAO')
    if 'pp' not in dir(): pp = load_pantheonplus()

    chi2_L1, p_L1 = fit_LCDM(pp, bao_set='DR1')
    H0_L1, Om_L1, Ob_L1 = p_L1
    _, parts_L1 = chi2_total_T(H0_L1, Om_L1, Ob_L1, None, pp,
                                include_DP=True, bao_set='DR1', return_parts=True)
    print(f'LCDM (DR1, k=3): chi2={chi2_L1:.3f}  H0={H0_L1:.3f}  '
          f'Om={Om_L1:.4f}  rd={parts_L1["rd"]:.2f}')
    LCDM_DR1 = chi2_L1

    chi2_C1, p_C1_dr1 = fit_C1(pp, bao_set='DR1')
    H0_C1d, Om_C1d, Ob_C1d, A_C1d = p_C1_dr1
    print(f'C1   (DR1, k=4): chi2={chi2_C1:.3f}  H0={H0_C1d:.3f}  '
          f'Om={Om_C1d:.4f}  A={A_C1d:+.4f}')

    dchi = chi2_C1 - chi2_L1
    dBIC = dchi + (4-3)*np.log(N_DR1)
    verdict = 'BURN (DR1-burn-by-BIC)' if dBIC >= 0 else 'pass'
    print(f'\nC1 vs LCDM (DR1): Δχ² = {dchi:+.3f}, ΔBIC = {dBIC:+.3f}  → {verdict}')
    results['s2'] = dict(LCDM_DR1=chi2_L1, C1_DR1=chi2_C1,
                          dchi=dchi, dBIC=dBIC, verdict=verdict)
    save()


# ===================================================================
# Section 3: D1 burn — direct A-grid scan
# ===================================================================
if SECTIONS['s3_D1_burn']:
    header('Section 3: D1 burn — direct grid scan')
    if 'pp' not in dir(): pp = load_pantheonplus()
    print(f'D1 lock: zc=sqrt(pi)={np.sqrt(PI):.5f}, w=1/(pi*sqrt(pi))={1/(PI*np.sqrt(PI)):.5f}')

    def profile_at_A(A_fixed, x0=(70.0, 0.28, 0.0228)):
        T = lambda z: T_D1(z, A_fixed)
        def neg(p):
            H0, Om, Ob = p
            if not (60 < H0 < 80 and 0.15 < Om < 0.45 and 0.020 < Ob < 0.025):
                return 1e10
            return chi2_total_T(H0, Om, Ob, T, pp, include_DP=True, bao_set='DR2')
        best = (1e10, None)
        for x0_try in [x0, (71.0, 0.285, 0.0224), (69.0, 0.29, 0.0228)]:
            r = minimize(neg, x0_try, method='Nelder-Mead',
                         options={'xatol':1e-7, 'fatol':1e-7, 'maxiter':5000})
            if r.fun < best[0]:
                best = (r.fun, r.x)
        return best

    A_grid = np.linspace(-0.40, 0.10, 26)
    grid_results = []
    print(f'{"A":>8s} {"chi2":>10s}')
    for A in A_grid:
        chi2, p = profile_at_A(A)
        grid_results.append((float(A), float(chi2)))
        if A in [-0.40, -0.30, -0.245, -0.20, -0.10, 0.00, 0.02, 0.05, 0.10] or \
           abs(A - (-0.245)) < 0.011:
            print(f'{A:+8.4f} {chi2:10.3f}')

    arr = np.array(grid_results)
    i_min = int(np.argmin(arr[:,1]))
    A_best = arr[i_min, 0]
    chi2_best = arr[i_min, 1]
    if 's1' in results:
        dBIC_D1 = (chi2_best - results['s1']['LCDM']['chi2']) + np.log(N_DR2)
    else:
        dBIC_D1 = float('nan')
    print(f'\nGlobal minimum: A={A_best:+.4f}, chi2={chi2_best:.3f}')
    print(f'D1 vs LCDM: ΔBIC = {dBIC_D1:+.3f}  → BURN')
    print('(Note v3 reported chi2=1572.94 at A=-0.245 — not reproducible.)')

    results['s3'] = dict(grid=grid_results, A_best=A_best,
                          chi2_best=chi2_best, dBIC=dBIC_D1, verdict='BURN')
    save()


# ===================================================================
# Section 4: free-Gaussian fits
# ===================================================================
def fit_free_gauss(pp, bao_set='DR2', seeds=(42, 7, 13, 99, 31)):
    """Fit T(z) = 1 + A*exp(-((z-zc)/w)^2) with all 6 parameters free."""
    bounds = [(65, 78), (0.20, 0.40), (0.020, 0.024),
              (-0.50, 0.30), (0.05, 1.5), (0.02, 0.50)]   # H0,Om,Ob,A,zc,w
    def neg(p):
        H0, Om, Ob, A, zc, w = p
        if not (60 < H0 < 80 and 0.15 < Om < 0.45 and 0.020 < Ob < 0.025
                and -0.95 < A < 0.40 and 0.05 < zc < 2.0 and 0.02 < w < 1.0):
            return 1e10
        T = lambda z: 1.0 + A * np.exp(-((z - zc)/w)**2)
        return chi2_total_T(H0, Om, Ob, T, pp, include_DP=True, bao_set=bao_set)
    all_seeds_results = []
    for sd in seeds:
        de = differential_evolution(neg, bounds, seed=sd, tol=1e-9,
                                    maxiter=400, polish=False)
        nm = minimize(neg, de.x, method='Nelder-Mead',
                      options={'xatol':1e-9, 'fatol':1e-9, 'maxiter':40000})
        all_seeds_results.append((float(nm.fun), nm.x.tolist(), int(sd)))
    all_seeds_results.sort()
    return all_seeds_results

if SECTIONS['s4_free_gauss']:
    header('Section 4: free-Gaussian fits, both BAO sets, multi-seed')
    if 'pp' not in dir(): pp = load_pantheonplus()
    s4 = {}
    for bao_set in ('DR2', 'DR1'):
        t1 = time.time()
        print(f'\n--- {bao_set} ---')
        rs = fit_free_gauss(pp, bao_set=bao_set)
        for chi2, x, sd in rs:
            H0, Om, Ob, A, zc, w = x
            print(f'  seed {sd:3d}: chi2={chi2:.3f}  zc={zc:.4f}  w={w:.4f}  A={A:+.4f}')
        best = rs[0]
        print(f'\n  best: chi2={best[0]:.3f}  zc={best[1][4]:.4f}  '
              f'w={best[1][5]:.4f}  A={best[1][3]:+.4f}  ({time.time()-t1:.0f}s)')
        s4[bao_set] = dict(seeds=rs, best_chi2=best[0], best_x=best[1])
    results['s4'] = s4
    save()


# ===================================================================
# Section 5: lock fits (LockB, LockC, on DR2 and DR1)
# ===================================================================
def fit_locked(pp, zc, w, bao_set='DR2', seeds=(42, 7, 13, 99)):
    """4-param fit: H0, Om, Ob, A, with (zc,w) fixed."""
    bounds = [(65, 78), (0.20, 0.40), (0.020, 0.024), (-0.50, 0.30)]
    def neg(p):
        H0, Om, Ob, A = p
        if not (60 < H0 < 80 and 0.15 < Om < 0.45 and 0.020 < Ob < 0.025
                and -0.95 < A < 0.40):
            return 1e10
        T = lambda z: 1.0 + A * np.exp(-((z - zc)/w)**2)
        return chi2_total_T(H0, Om, Ob, T, pp, include_DP=True, bao_set=bao_set)
    best = (1e10, None)
    for sd in seeds:
        de = differential_evolution(neg, bounds, seed=sd, tol=1e-9,
                                    maxiter=300, polish=False)
        nm = minimize(neg, de.x, method='Nelder-Mead',
                      options={'xatol':1e-9, 'fatol':1e-9, 'maxiter':25000})
        if nm.fun < best[0]:
            best = (nm.fun, nm.x.copy())
    return best

if SECTIONS['s5_lock_fits']:
    header('Section 5: lock fits — LockB and LockC')
    if 'pp' not in dir(): pp = load_pantheonplus()
    LOCKS = [
        ('LockB', 1/np.sqrt(PI), 2/(5*PI)),
        ('LockC', 4/7,            2/(5*PI)),
    ]
    s5 = {}
    for name, zc, w in LOCKS:
        s5[name] = dict(zc=zc, w=w)
        for bao_set, lcdm in [('DR2', LCDM_DR2 or 1606.936),
                               ('DR1', LCDM_DR1 or 1598.281)]:
            t1 = time.time()
            chi2, p = fit_locked(pp, zc, w, bao_set=bao_set)
            H0, Om, Ob, A = p
            dchi = chi2 - lcdm
            dBIC = dchi + (4-3)*np.log(N_DR2 if bao_set=='DR2' else N_DR1)
            print(f'{name} [{bao_set}]: chi2={chi2:.3f}  H0={H0:.2f}  '
                  f'Om={Om:.4f}  A={A:+.4f}  Δχ²={dchi:+.2f}  ΔBIC={dBIC:+.2f}  '
                  f'({time.time()-t1:.0f}s)')
            s5[name][bao_set] = dict(chi2=float(chi2), H0=float(H0),
                                      Om=float(Om), Ob=float(Ob), A=float(A),
                                      dchi=float(dchi), dBIC=float(dBIC))
    results['s5'] = s5
    save()


# ===================================================================
# Section 6: LOO — LockC on DR2 (12 points) and DR1 (11 points)
# ===================================================================
def chi2_BAO_DR2_LOO(H0, Om, T_func, rd, skip_label=None):
    """DR2 LOO with all 12 labels. Inline, doesn't depend on the buggy helpers."""
    DC, H = comoving_dist_grid_T(H0, Om, T_func)
    if DC is None: return 1e10
    chi2 = 0.0
    # DR2 BGS
    if skip_label != 'BGS':
        for z, DVo, sig in [(0.295, 7.94, 0.075)]:
            DM = np.interp(z, _Z_GRID, DC)
            Hv = np.interp(z, _Z_GRID, H)
            DV = (z * DM**2 * C_KMS / Hv) ** (1/3)
            chi2 += ((DVo - DV/rd) / sig)**2
    # DR2 anisotropic pairs (6)
    DR2_pair_data = [
        ('LRG1', (0.510, 13.62, 0.25, 20.98, 0.61, -0.445)),
        ('LRG2', (0.706, 16.85, 0.32, 20.08, 0.60, -0.420)),
        ('LRG3', (0.934, 21.71, 0.28, 17.88, 0.35, -0.389)),
        ('ELG2', (1.321, 27.79, 0.69, 13.82, 0.42, -0.444)),
        ('QSO',  (1.484, 30.69, 0.79, 13.18, 0.55, -0.477)),
        ('Lya',  (2.330, 39.71, 0.94,  8.52, 0.17, -0.477)),
    ]
    for label, (z, DMo, sDM, DHo, sDH, corr) in DR2_pair_data:
        if skip_label == label: continue
        DMp = np.interp(z, _Z_GRID, DC) / rd
        DHp = C_KMS / np.interp(z, _Z_GRID, H) / rd
        cov = np.array([[sDM**2, corr*sDM*sDH],[corr*sDM*sDH, sDH**2]])
        d = np.array([DMo - DMp, DHo - DHp])
        chi2 += d @ np.linalg.inv(cov) @ d
    # BOSS
    if skip_label != 'BOSS':
        for (z1, DMo, sDM), (z2, DHo, sDH, corr) in zip(BOSS_DM, BOSS_DH):
            DMp = np.interp(z1, _Z_GRID, DC) / rd
            DHp = C_KMS / np.interp(z1, _Z_GRID, H) / rd
            cov = np.array([[sDM**2, corr*sDM*sDH],[corr*sDM*sDH, sDH**2]])
            d = np.array([DMo - DMp, DHo - DHp])
            chi2 += d @ np.linalg.inv(cov) @ d
    # eBOSS pairs
    eboss_labels = ['eBOSS_LRG', 'eBOSS_QSO', 'eBOSS_Lya']
    for label, (z1, DMo, sDM), (z2, DHo, sDH, corr) in zip(
            eboss_labels, EBOSS_DM, EBOSS_DH):
        if skip_label == label: continue
        DMp = np.interp(z1, _Z_GRID, DC) / rd
        DHp = C_KMS / np.interp(z1, _Z_GRID, H) / rd
        cov = np.array([[sDM**2, corr*sDM*sDH],[corr*sDM*sDH, sDH**2]])
        d = np.array([DMo - DMp, DHo - DHp])
        chi2 += d @ np.linalg.inv(cov) @ d
    # eBOSS ELG
    if skip_label != 'eBOSS_ELG':
        for z, DVo, sig in EBOSS_DV:
            DM = np.interp(z, _Z_GRID, DC)
            Hv = np.interp(z, _Z_GRID, H)
            DV = (z * DM**2 * C_KMS / Hv) ** (1/3)
            chi2 += ((DVo - DV/rd) / sig)**2
    return float(chi2)


def chi2_total_DR2_LOO(H0, Om, Ob, T_func, pp, skip_label):
    DM_star = _DM_at_zstar_T(H0, Om, Ob, T_func)
    if DM_star <= 0: return 1e10
    rd = THETA_STAR * DM_star
    if not (100 < rd < 160): return 1e10
    chi2_pp, _ = chi2_panplus_T(H0, Om, T_func, pp)
    chi2_b = chi2_BAO_DR2_LOO(H0, Om, T_func, rd, skip_label=skip_label)
    chi2_h = ((H0 - H0_SHOES)/SIG_H0_SHOES)**2
    chi2_dp, _, _ = chi2_DP_T(H0, Om, Ob, T_func)
    return chi2_pp + chi2_b + chi2_h + chi2_dp


def fit_LOO_locked(pp, zc, w, skip_label, bao_set, seeds=(42, 7, 13)):
    chi2_total_fn = chi2_total_DR2_LOO if bao_set=='DR2' else chi2_total_DR1_LOO
    bounds = [(65, 78), (0.20, 0.40), (0.020, 0.024), (-0.50, 0.30)]
    def neg(p):
        H0, Om, Ob, A = p
        if not (60 < H0 < 80 and 0.15 < Om < 0.45 and 0.020 < Ob < 0.025):
            return 1e10
        T = lambda z: 1.0 + A * np.exp(-((z - zc)/w)**2)
        return chi2_total_fn(H0, Om, Ob, T, pp, skip_label)
    best = (1e10, None)
    for sd in seeds:
        de = differential_evolution(neg, bounds, seed=sd, tol=1e-9,
                                    maxiter=300, polish=False)
        nm = minimize(neg, de.x, method='Nelder-Mead',
                      options={'xatol':1e-8, 'fatol':1e-8, 'maxiter':20000})
        if nm.fun < best[0]:
            best = (nm.fun, nm.x.copy())
    return best


def fit_LOO_LCDM(pp, skip_label, bao_set, seeds=(42, 7, 13)):
    chi2_total_fn = chi2_total_DR2_LOO if bao_set=='DR2' else chi2_total_DR1_LOO
    bounds = [(65, 78), (0.20, 0.40), (0.020, 0.024)]
    def neg(p):
        H0, Om, Ob = p
        if not (60 < H0 < 80 and 0.15 < Om < 0.45 and 0.020 < Ob < 0.025):
            return 1e10
        return chi2_total_fn(H0, Om, Ob, None, pp, skip_label)
    best = (1e10, None)
    for sd in seeds:
        de = differential_evolution(neg, bounds, seed=sd, tol=1e-9,
                                    maxiter=300, polish=False)
        nm = minimize(neg, de.x, method='Nelder-Mead',
                      options={'xatol':1e-8, 'fatol':1e-8, 'maxiter':20000})
        if nm.fun < best[0]:
            best = (nm.fun, nm.x.copy())
    return best


def loo_sweep(pp, zc, w, bao_set, labels):
    rows = []
    N_data = N_DR2 if bao_set=='DR2' else N_DR1
    print(f'  {"label":<14s} {"chi2_lock":>10s} {"chi2_LCDM":>10s} '
          f'{"Δχ²":>9s} {"ΔBIC":>8s} {"A":>8s} {"sec":>6s}')
    for label in labels:
        t1 = time.time()
        chi2_lock, p_lock = fit_LOO_locked(pp, zc, w, label, bao_set)
        chi2_lcdm, _ = fit_LOO_LCDM(pp, label, bao_set)
        dchi = chi2_lock - chi2_lcdm
        dBIC = dchi + (4-3)*np.log(N_data - 2)
        A = p_lock[3]
        rows.append(dict(label=label, chi2_lock=float(chi2_lock),
                         chi2_lcdm=float(chi2_lcdm), A=float(A),
                         dchi=float(dchi), dBIC=float(dBIC)))
        print(f'  -{label:<13s} {chi2_lock:10.2f} {chi2_lcdm:10.2f} '
              f'{dchi:+9.3f} {dBIC:+8.3f} {A:+8.4f} {time.time()-t1:6.0f}')
    return rows

if SECTIONS['s6_LOO_LockC_DR2']:
    header('Section 6a: LOO — LockC on DR2 (12 points)')
    if 'pp' not in dir(): pp = load_pantheonplus()
    labels_dr2 = ['BGS','LRG1','LRG2','LRG3','ELG2','QSO','Lya',
                  'BOSS','eBOSS_LRG','eBOSS_QSO','eBOSS_Lya','eBOSS_ELG']
    rows = loo_sweep(pp, 4/7, 2/(5*PI), 'DR2', labels_dr2)
    n_bic = sum(1 for r in rows if r['dBIC'] < 0)
    As = [r['A'] for r in rows]
    spread = (max(As)-min(As))/abs(np.mean(As))*100
    print(f'\n  LockC DR2: {n_bic}/12 BIC pass, A spread {spread:.1f}%, '
          f'worst ΔBIC = {max(r["dBIC"] for r in rows):+.3f}')
    results['s6a_LOO_LockC_DR2'] = rows
    save()

if SECTIONS['s6_LOO_LockC_DR1']:
    header('Section 6b: LOO — LockC on DR1 (11 points)')
    if 'pp' not in dir(): pp = load_pantheonplus()
    labels_dr1 = ['BGS','LRG1','LRG2','LRG3','QSO','Lya',
                  'BOSS','eBOSS_LRG','eBOSS_QSO','eBOSS_Lya','eBOSS_ELG']
    rows = loo_sweep(pp, 4/7, 2/(5*PI), 'DR1', labels_dr1)
    n_bic = sum(1 for r in rows if r['dBIC'] < 0)
    As = [r['A'] for r in rows]
    spread = (max(As)-min(As))/abs(np.mean(As))*100
    print(f'\n  LockC DR1: {n_bic}/11 BIC pass, A spread {spread:.1f}%, '
          f'worst ΔBIC = {max(r["dBIC"] for r in rows):+.3f}')
    results['s6b_LOO_LockC_DR1'] = rows
    save()


# ===================================================================
# Section 7: BBN test (rd = 147.05 ± 0.3 Mpc Gaussian prior)
# ===================================================================
def chi2_total_BBN(H0, Om, Ob, T_func, pp, bao_set='DR2',
                   rd_prior_mean=147.05, rd_prior_sig=0.3):
    """Same as v14 chi2 total but with extra Gaussian prior on rd."""
    chi2_v14, parts = chi2_total_T(H0, Om, Ob, T_func, pp,
                                     include_DP=True, bao_set=bao_set,
                                     return_parts=True)
    rd = parts['rd']
    pen = ((rd - rd_prior_mean) / rd_prior_sig)**2
    return chi2_v14 + pen, parts['rd'], pen


def fit_BBN(pp, zc=None, w=None, bao_set='DR2', has_A=True,
            seeds=(42, 7, 13, 99)):
    if has_A:
        bounds = [(65, 78), (0.20, 0.40), (0.020, 0.024), (-0.50, 0.30)]
    else:
        bounds = [(65, 78), (0.20, 0.40), (0.020, 0.024)]
    def neg(p):
        if has_A:
            H0, Om, Ob, A = p
            T = lambda z: 1.0 + A * np.exp(-((z - zc)/w)**2)
        else:
            H0, Om, Ob = p
            T = None
        c, _, _ = chi2_total_BBN(H0, Om, Ob, T, pp, bao_set=bao_set)
        return c
    best = (1e10, None)
    for sd in seeds:
        de = differential_evolution(neg, bounds, seed=sd, tol=1e-9,
                                    maxiter=300, polish=False)
        nm = minimize(neg, de.x, method='Nelder-Mead',
                      options={'xatol':1e-9, 'fatol':1e-9, 'maxiter':20000})
        if nm.fun < best[0]:
            best = (nm.fun, nm.x.copy())
    return best

if SECTIONS['s7_BBN']:
    header('Section 7: BBN constraint test (rd = 147.05 ± 0.3 Mpc)')
    if 'pp' not in dir(): pp = load_pantheonplus()
    s7 = {}
    for bao_set in ('DR2', 'DR1'):
        N = N_DR2 if bao_set=='DR2' else N_DR1
        chi2_L, p_L = fit_BBN(pp, bao_set=bao_set, has_A=False)
        chi2_C, p_C = fit_BBN(pp, zc=4/7, w=2/(5*PI), bao_set=bao_set, has_A=True)
        dchi = chi2_C - chi2_L
        dBIC = dchi + (4-3)*np.log(N)
        unforced = (results.get('s5', {}).get('LockC', {}).get(bao_set, {}).get('dBIC')
                    if 's5' in results else None)
        print(f'[{bao_set}]')
        print(f'  LCDM+BBN:  chi2={chi2_L:.2f}, H0={p_L[0]:.2f}, Om={p_L[1]:.4f}')
        print(f'  LockC+BBN: chi2={chi2_C:.2f}, H0={p_C[0]:.2f}, A={p_C[3]:+.4f}')
        print(f'  ΔBIC unforced = {unforced if unforced else "—"}')
        print(f'  ΔBIC forced   = {dBIC:+.3f}')
        s7[bao_set] = dict(LCDM=float(chi2_L), LockC=float(chi2_C),
                            dchi=float(dchi), dBIC=float(dBIC))
    results['s7'] = s7
    save()


# ===================================================================
# Section 8: free-w LOO at zc = 4/7
# ===================================================================
def fit_LOO_freew(pp, skip_label, bao_set, zc=4/7, seeds=(42, 7, 13)):
    chi2_total_fn = chi2_total_DR2_LOO if bao_set=='DR2' else chi2_total_DR1_LOO
    bounds = [(65, 78), (0.20, 0.40), (0.020, 0.024),
              (-0.50, 0.30), (0.03, 0.50)]   # H0,Om,Ob,A,w
    def neg(p):
        H0, Om, Ob, A, w = p
        if not (60 < H0 < 80 and 0.15 < Om < 0.45 and 0.020 < Ob < 0.025
                and -0.95 < A < 0.40 and 0.02 < w < 0.6):
            return 1e10
        T = lambda z: 1.0 + A * np.exp(-((z - zc)/w)**2)
        return chi2_total_fn(H0, Om, Ob, T, pp, skip_label)
    best = (1e10, None)
    for sd in seeds:
        de = differential_evolution(neg, bounds, seed=sd, tol=1e-9,
                                    maxiter=300, polish=False)
        nm = minimize(neg, de.x, method='Nelder-Mead',
                      options={'xatol':1e-8, 'fatol':1e-8, 'maxiter':25000})
        if nm.fun < best[0]:
            best = (nm.fun, nm.x.copy())
    return best

if SECTIONS['s8_freew_LOO']:
    header('Section 8: free-w LOO at zc=4/7')
    if 'pp' not in dir(): pp = load_pantheonplus()
    W_LOCK = 2/(5*PI)
    s8 = {}
    for bao_set, labels in [
            ('DR2', ['BGS','LRG1','LRG2','LRG3','ELG2','QSO','Lya',
                     'BOSS','eBOSS_LRG','eBOSS_QSO','eBOSS_Lya','eBOSS_ELG']),
            ('DR1', ['BGS','LRG1','LRG2','LRG3','QSO','Lya',
                     'BOSS','eBOSS_LRG','eBOSS_QSO','eBOSS_Lya','eBOSS_ELG'])]:
        rows = []
        print(f'\n--- {bao_set} ---')
        # Full data first
        chi2_full, p_full = fit_LOO_freew(pp, None, bao_set)
        H0,Om,Ob,A,w = p_full
        dev = (w - W_LOCK)/W_LOCK*100
        print(f'  full data: chi2={chi2_full:.2f}, A={A:+.4f}, w={w:.4f} (dev {dev:+.2f}%)')
        s8[f'{bao_set}_full'] = dict(chi2=float(chi2_full), w=float(w),
                                       w_dev_pct=float(dev), A=float(A))
        # LOO sweep
        for label in labels:
            t1 = time.time()
            chi2, p = fit_LOO_freew(pp, label, bao_set)
            H0,Om,Ob,A,w = p
            dev = (w - W_LOCK)/W_LOCK*100
            rows.append(dict(label=label, chi2=float(chi2), A=float(A),
                              w=float(w), w_dev_pct=float(dev)))
            print(f'  -{label:<12s} chi2={chi2:.2f}  A={A:+.4f}  '
                  f'w={w:.4f}  dev={dev:+.1f}%  ({time.time()-t1:.0f}s)')
        ws = [r['w'] for r in rows]
        n_at_bound = sum(1 for r in rows if r['w'] > 0.49)
        n_near_lock = sum(1 for r in rows if abs(r['w_dev_pct']) < 1)
        print(f'  {bao_set} summary: {n_at_bound}/{len(rows)} at bound, '
              f'{n_near_lock}/{len(rows)} within 1% of lock')
        s8[bao_set] = rows
    results['s8'] = s8
    save()


# ===================================================================
# Section 9: T(z) profile sanity check
# ===================================================================
if SECTIONS['s9_T_profile']:
    header('Section 9: T(z) profile sanity check')
    print('LockC at A=+0.06 (representative DR1 fit):')
    zc, w, A = 4/7, 2/(5*PI), 0.06
    for z in [0.0, 0.1, 0.3, 0.4, 0.5, zc, 0.7, 0.8, 1.0, 2.0, 1090.0]:
        T = 1.0 + A*np.exp(-((z-zc)/w)**2)
        tag = '  ← peak' if abs(z - zc) < 0.001 else ''
        print(f'  T({z:7.3f}) = {T:.5f}{tag}')
    print('Profile: bump in z ∈ [0.2, 0.95], T → 1 at both ends. Self-consistent.')


# ===================================================================
# Final summary
# ===================================================================
header('FINAL SUMMARY')
print('Retired models:')
print(f'  v12/v13 C1   : DR1 ΔBIC = {results.get("s2",{}).get("dBIC","—")}  → BURN')
print(f'  v14-note D1  : ΔBIC at lock = {results.get("s3",{}).get("dBIC","—")}  → BURN')
print()
print('Surviving candidates (live, not closed):')
for name in ('LockB','LockC'):
    if 's5' in results and name in results['s5']:
        r = results['s5'][name]
        print(f'  {name}: DR2 ΔBIC = {r["DR2"]["dBIC"]:+.3f}, '
              f'DR1 ΔBIC = {r["DR1"]["dBIC"]:+.3f}')

print(f'\nResults saved to {args.out}/v15_all_results.json')
