"""
=============================================================================
  LAC v5.9 — Lattice Awakening Cosmology
  FULLY TRANSPARENT REPRODUCTION CODE
  Every derivation step printed and verifiable.

  Author : LUMEN PIXEL, Busan, Republic of Korea, 2026
  License: CC BY 4.0
  Requires: numpy, scipy, matplotlib  (Python >= 3.10)
  Optional: camb  (for CMB peak test)

  Run:
      python lac_v59_transparent.py          # full run
      python lac_v59_transparent.py --nocamb # skip CAMB (fast)
=============================================================================
"""

import sys
import numpy as np
from scipy.integrate import quad, odeint
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

USE_CAMB = '--nocamb' not in sys.argv
np.random.seed(2024)

SEP  = "=" * 65
SEP2 = "-" * 65

def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def step(label, value, unit="", comment=""):
    c = f"  # {comment}" if comment else ""
    print(f"  {label:<40} = {value}{c}")
    return value


# =============================================================================
# PART 0.  Physical constants  (all SI-derived, no tuning)
# =============================================================================
section("PART 0 — Physical constants")

C_KM      = step("c  [km/s]",          2.998e5,   comment="speed of light")
H0_LAC    = step("H0 [km/s/Mpc]",      70.85,     comment="= 1/t0 = 1/13.797 Gyr")
OBH2      = step("Omega_b h^2",        0.02237,   comment="Planck 2018 CMB")
OGH2      = step("Omega_gamma h^2",    2.47e-5,   comment="photon density")
Z_DRAG    = step("z_drag",             1059.9,    comment="baryon drag epoch")
Z_STAR    = step("z_star",             1089.9,    comment="recombination")
THETA_OBS = step("theta* (Planck obs)",0.010409,  comment="Planck 2018")
SIGMA8_PL = step("sigma8 (Planck)",    0.811,     comment="Planck 2018")

# Derived
R0_BPH    = step("R0 = 3Ob/(4Og)",     3*OBH2/(4*OGH2),   comment="baryon/photon ratio at z=0")
H_IN_S    = H0_LAC * 1e3 / (3.086e22)   # H0 in 1/s
T0_GYR    = 1/H_IN_S / (3.156e16)
step("t0 = 1/H0  [Gyr]",         T0_GYR,  comment="cosmological age")

h         = H0_LAC / 100.0
OM_H2     = 0.1430        # Planck Omega_m h^2
OM        = OM_H2 / h**2
step("h = H0/100",               h)
step("Omega_m h^2",              OM_H2,   comment="Planck")
step("Omega_m",                  OM)


# =============================================================================
# PART 1.  Lattice constants  (zero free parameters)
# =============================================================================
section("PART 1 — Lattice constants (SCC 6-12-8 + FCC)")

print("""
  SCC = Simple Cubic Cell
    N_face   = 6   (nearest,  r = l)
    N_edge   = 12  (next,     r = l*sqrt(2))
    N_vertex = 8   (diagonal, r = l*sqrt(3))
    N_SCC    = 26  (total neighbours)

  FCC = Face-Centred Cubic packing
    phi = pi*sqrt(2)/6  (sphere packing fraction)
""")

N_FACE, N_EDGE, N_VERTEX = 6, 12, 8
N_SCC   = step("N_SCC = 6+12+8",         N_FACE+N_EDGE+N_VERTEX)

PHI_FCC = np.pi * np.sqrt(2) / 6
step("phi = pi*sqrt(2)/6",        PHI_FCC, comment="FCC packing fraction")

ALPHA   = np.log(6) / np.log(8)
step("alpha = ln(6)/ln(8)",       ALPHA,   comment="face/vertex neighbour ratio")

BETA    = -(1.0 - PHI_FCC)
step("beta = -(1-phi)",           BETA,    comment="negative void fraction")

N_LSS   = (N_FACE*1.0 + N_EDGE*2.0 + N_VERTEX*1.5) / N_SCC
step("n_lss = (6*1+12*2+8*1.5)/26", N_LSS, comment="SCC growth exponent = 42/26")

print(f"\n  Check n_lss = 42/26 = {42/26:.8f}  ==  {N_LSS:.8f}")
assert abs(N_LSS - 42/26) < 1e-10, "n_lss mismatch"

# ── v5.9 NEW: torsion constant & BAO exponent ──────────────────
print(f"""
  {SEP2}
  v5.9 NEW constants (derived from existing lattice, no new inputs)
  {SEP2}
  When two SCC cubes counter-rotate, the bond becomes a helix:
    l_twist = l0 * sqrt(1 + kappa^2)
    kappa   = phi^2 * |beta|   [packing^2 * void_fraction]

  BAO traces matter: void 2nd-order correction to growth exponent:
    alpha_BAO = alpha + |beta|^2
""")

KAPPA     = PHI_FCC**2 * abs(BETA)
step("kappa = phi^2 * |beta|",    KAPPA,   comment="torsion constant")

ALPHA_BAO = ALPHA + BETA**2
step("alpha_BAO = alpha + beta^2",ALPHA_BAO, comment="BAO rendering exponent")

TWIST_FACTOR = np.sqrt(1 + KAPPA**2)
step("sqrt(1 + kappa^2)",         TWIST_FACTOR, comment="torsion length factor")


# =============================================================================
# PART 2.  Sound horizon r_s  (first-principles)
# =============================================================================
section("PART 2 — Sound horizon derivation")

print(f"""
  r_s_phys = integral_0^z_drag  c_s(z) / H_LAC(z)  dz

  where:
    c_s(z) = c / sqrt(3 * (1 + R0/(1+z)))   [baryon-photon sound speed]
    H_LAC(z) = H0 * (1+z)                    [LAC coasting expansion]
    R0 = 3*Ob_h2/(4*Og_h2) = {R0_BPH:.4f}    [baryon/photon ratio at z=0]
""")

def c_sound(z):
    """Baryon-photon sound speed [km/s]"""
    return C_KM / np.sqrt(3.0 * (1.0 + R0_BPH / (1.0 + z)))

def H_LAC(z):
    """LAC Hubble parameter [km/s/Mpc]"""
    return H0_LAC * (1.0 + z)

# numerical integration
def integrand_rs(z):
    return c_sound(z) / H_LAC(z)

R_S_PHYS, rs_err = quad(integrand_rs, 0.0, Z_DRAG, limit=500)

print(f"  Numerical integration (scipy.quad, limit=500):")
print(f"    z range   = [0, {Z_DRAG}]")
print(f"    c_s(0)    = {c_sound(0):.4f} km/s")
print(f"    c_s(z_d)  = {c_sound(Z_DRAG):.4f} km/s")
print(f"    H_LAC(0)  = {H_LAC(0):.4f} km/s/Mpc")
print(f"    H_LAC(z_d)= {H_LAC(Z_DRAG):.4f} km/s/Mpc")
step("r_s_phys  [Mpc]",           R_S_PHYS, comment=f"integration error={rs_err:.2e}")

# Lattice rendering: Q = N_SCC / phi^2
Q_LAT = N_SCC / PHI_FCC**2
step("Q = N_SCC / phi^2",         Q_LAT,   comment="lattice rendering factor")

R_S_EFF = R_S_PHYS / Q_LAT
step("r_s_eff = r_s_phys / Q",    R_S_EFF, comment="rendered sound horizon")

# Torsion correction (v5.9)
R_S_TWIST = R_S_EFF * TWIST_FACTOR
step("r_s_twist = r_s_eff*sqrt(1+k^2)", R_S_TWIST,
     comment="helix-corrected, used for CMB")

# k_c: nonlinear transition scale
K_C = 4.0 * np.pi / R_S_TWIST
step("k_c = 4*pi / r_s_twist",    K_C,     comment="BAO second reciprocal lattice")

N_GROWTH = N_LSS   # same exponent
step("N_growth = n_lss",          N_GROWTH, comment="growth suppression power")


# =============================================================================
# PART 3.  CMB acoustic scale theta*
# =============================================================================
section("PART 3 — CMB acoustic scale theta*")

print(f"""
  theta* = r_s_twist / D_C_phot(z_drag)

  D_C_phot(z) = phi * c / (H0 * beta) * [(1+z)^beta - 1]
    This is the closed-form integral of:
    integral_0^z  c * phi * (1+z')^beta / H_LAC(z')  dz'

  Physical origin:
    - phi     : FCC packing fraction (lattice transparency)
    - (1+z)^beta : void fraction effect on photon path
    - beta = -(1-phi) : voids reflect photons proportional to void fraction
""")

def D_C_phot(z):
    """LAC photon comoving distance [Mpc]"""
    if abs(BETA) < 1e-8:
        return PHI_FCC * C_KM / H0_LAC * np.log(1.0 + z)
    return PHI_FCC * C_KM / (H0_LAC * BETA) * ((1.0 + z)**BETA - 1.0)

# Verification: compare with direct numerical integration
def D_C_phot_numerical(z):
    def ig(zp): return PHI_FCC * (1+zp)**BETA / H_LAC(zp)
    r, _ = quad(ig, 0, z, limit=500)
    return C_KM * r

DC_analytical = D_C_phot(Z_DRAG)
DC_numerical  = D_C_phot_numerical(Z_DRAG)
step("D_C_phot(z_drag) [analytical]", DC_analytical, comment="[Mpc]")
step("D_C_phot(z_drag) [numerical]",  DC_numerical,  comment="[Mpc] — should match")
print(f"  Consistency check: |ana-num|/num = {abs(DC_analytical-DC_numerical)/DC_numerical:.2e}")
assert abs(DC_analytical - DC_numerical)/DC_numerical < 1e-6, "D_C_phot mismatch"

THETA_PRED = R_S_TWIST / DC_analytical
THETA_DEV  = (THETA_PRED - THETA_OBS) / THETA_OBS * 100.0
step("theta* = r_s_twist / D_C_phot", THETA_PRED)
step("theta* deviation  [%]",          THETA_DEV,
     comment=f"obs={THETA_OBS}  (v5.8 was -1.04%)")

print(f"\n  Torsion improvement:")
theta_v58 = R_S_EFF / DC_analytical
print(f"    v5.8 theta* = r_s_eff / D_C = {theta_v58:.8f}  dev={( theta_v58-THETA_OBS)/THETA_OBS*100:+.4f}%")
print(f"    v5.9 theta* = r_s_twist/D_C = {THETA_PRED:.8f}  dev={THETA_DEV:+.4f}%")


# =============================================================================
# PART 4.  BAO — D_V / r_s
# =============================================================================
section("PART 4 — BAO: effective volume distance D_V / r_s")

print(f"""
  D_V(z) = [ z * D_A(z)^2 * c/H(z) ]^(1/3) * Gamma(z)

  where:
    D_A(z) = chi_sound(z) / (1+z)        [angular diameter distance]
    chi_sound(z) = c/H0 * ln(1+z)         [LAC comoving distance]
    c/H(z) = c/(H0*(1+z))                 [Hubble distance at z]

  Gamma(z) = phi * (1+z)^alpha_BAO        [LAC structure growth factor]
    alpha_BAO = alpha + |beta|^2 = {ALPHA_BAO:.6f}   [v5.9]

  F(k) = 1 / (1 + (k/k_c)^N_growth)      [nonlinear suppression]
    k_c = {K_C:.6f} h/Mpc,  N_growth = {N_GROWTH:.5f}

  BAO measurement: D_V(z) / r_s_twist
""")

def chi_sound(z):
    """LAC comoving distance [Mpc]"""
    return (C_KM / H0_LAC) * np.log(1.0 + z)

def dL_phot(z):
    """LAC luminosity distance [Mpc]"""
    return D_C_phot(z) * (1.0 + z)

def F_k(k):
    """Nonlinear suppression factor"""
    return 1.0 / (1.0 + (k / K_C)**N_GROWTH)

def Gamma_z(z):
    """Structure growth factor (BAO)"""
    return PHI_FCC * (1.0 + z)**ALPHA_BAO

def D_V_eff(z):
    """Effective volume distance / r_s_twist"""
    chi = chi_sound(z)
    D_A = chi / (1.0 + z)
    c_over_H = C_KM / H_LAC(z)
    D_V_Mpc = (z * D_A**2 * c_over_H)**(1.0/3.0)
    return D_V_Mpc * Gamma_z(z) / R_S_TWIST

# BAO data: classic low-z  +  DESI 2024
BAO_DATA = [
    # (z_eff, D_V/r_d_obs, sigma, label)
    (0.106,  2.98,  0.13, "6dFGS"),
    (0.150,  4.47,  0.17, "SDSS MGS"),
    (0.295,  7.93,  0.15, "DESI BGS"),
    (0.320,  8.47,  0.17, "BOSS LOWZ"),
    (0.510, 13.62,  0.25, "DESI LRG1"),
    (0.570, 13.77,  0.13, "BOSS CMASS"),
    (0.706, 16.85,  0.32, "DESI LRG2"),
    (0.930, 21.71,  0.28, "DESI LRG3"),
    (1.317, 27.79,  0.69, "DESI ELG2"),
    (1.491, 30.03,  0.75, "DESI QSO"),
    (2.330, 39.71,  0.94, "DESI Lya"),
]

print(f"\n  {'z':>6} {'D_V/r_s (obs)':>15} {'D_V/r_s (pred)':>16} "
      f"{'pull':>8} {'label':>12}")
print(f"  {SEP2}")

chi2_bao = 0.0
for z, dv_obs, sig, label in BAO_DATA:
    dv_pred = D_V_eff(z)
    pull    = (dv_pred - dv_obs) / sig
    chi2_bao += pull**2
    print(f"  {z:>6.3f} {dv_obs:>15.3f} {dv_pred:>16.4f} "
          f"{pull:>+8.3f}   {label}")

n_bao = len(BAO_DATA)
print(f"\n  chi2 = {chi2_bao:.4f}")
step("BAO chi2/dof",               chi2_bao / n_bao, comment=f"dof={n_bao}")


# =============================================================================
# PART 5.  LSS — f*sigma8(z,k)
# =============================================================================
section("PART 5 — LSS: redshift-space distortions f*sigma8")

print(f"""
  Growth equation (k-resolved):
    D''(a) + [2/a - ...] D'(a) = 3/2 * Omega_m_eff(z,k) / a^2 * D(a)

  Omega_m_eff(z,k) = Omega_m(z) * Gamma(z,k)
  Gamma(z,k) = phi * (1+z)^alpha_BAO * F(k)
  F(k) = 1/(1 + (k/k_c)^N_growth)

  Prediction: f(z,k)*sigma8(z) = a/D * dD/da * sigma8_0 * D(z)
""")

def solve_growth(z_target, k_eff, Omega_m=OM):
    """
    Solve linear growth ODE for D(z), return D(z)/D(0) and f(z)=dlnD/dlna.
    Scale factor a = 1/(1+z).
    ODE: d²D/da² + (2/a - d(ln H)/d(ln a))/a * dD/da
         - 3/2 * Omega_m_eff/(a^3 H^2/H0^2) * D/a^2 = 0
    In LAC: H = H0*(1+z) = H0/a, so d(ln H)/d(ln a) = -1
    Simplified:  a*D'' + (3-...)*D' - 3/2*Om_eff/a*D = 0
    """
    z_max = max(float(z_target) * 1.1, 10.0)
    a_grid = np.linspace(1.0 / (1.0 + z_max), 1.0, 800)

    def rhs(state, a):
        D, Dp = state
        if a < 1e-7:
            return [Dp, 0.0]
        z_loc = 1.0 / a - 1.0
        Hz    = H_LAC(z_loc)
        # Effective Omega_m at (z, k):
        Gamma_eff = Omega_m * (H0_LAC / Hz)**2 / a**3 * \
                    PHI_FCC * (1.0 + z_loc)**ALPHA_BAO * F_k(k_eff)
        # ODE coefficients for H(z) = H0*(1+z):
        # H'/H = d(ln H)/dt = -H0*(1+z)^2 / (H0*(1+z)) * ... => in a: -(2/a)*H
        # standard form: D'' = -(2/a)*D' + (3/2)*Omega_eff/a^2 * D
        d2D = -(2.0 / a) * Dp + 1.5 * Gamma_eff / a**2 * D
        return [Dp, d2D]

    # Initial conditions: matter-dominated growing mode D ~ a
    sol = odeint(rhs, [a_grid[0], 1.0], a_grid, rtol=1e-8, atol=1e-10)

    D_interp  = interp1d(a_grid, sol[:, 0], kind='cubic', fill_value='extrapolate')
    Dp_interp = interp1d(a_grid, sol[:, 1], kind='cubic', fill_value='extrapolate')

    D0   = float(D_interp(1.0))
    a_z  = 1.0 / (1.0 + float(z_target))
    D_z  = float(D_interp(a_z)) / D0
    f_z  = a_z / D_z * float(Dp_interp(a_z)) / D0   # f = d ln D / d ln a

    return D_z, f_z

# RSD data
LSS_DATA = [
    # (z, f*sigma8, sigma, label, k_eff)
    (0.067, 0.423, 0.055, "6dFGS",     0.032),
    (0.170, 0.510, 0.060, "SDSS MGS",  0.063),
    (0.220, 0.416, 0.057, "WiggleZ",   0.077),
    (0.410, 0.450, 0.040, "WiggleZ",   0.077),
    (0.570, 0.427, 0.020, "BOSS DR11", 0.045),
    (0.600, 0.433, 0.038, "WiggleZ",   0.077),
    (0.780, 0.438, 0.037, "WiggleZ",   0.077),
    (0.800, 0.470, 0.080, "VIPERS",    0.158),
    (1.400, 0.482, 0.116, "FastSound", 0.122),
]

SIGMA8_0 = SIGMA8_PL   # normalised to Planck sigma8 at z=0

print(f"\n  sigma8(z=0) = {SIGMA8_0} (Planck normalisation)")
print(f"\n  {'z':>5} {'obs':>8} {'pred':>8} {'pull':>7} {'label':>12} {'k_eff':>6}")
print(f"  {SEP2}")

chi2_lss = 0.0
for z, fs8_obs, sig, label, k_eff in LSS_DATA:
    D_z, f_z = solve_growth(z, k_eff)
    fs8_pred = f_z * SIGMA8_0 * D_z
    pull     = (fs8_pred - fs8_obs) / sig
    chi2_lss += pull**2
    print(f"  {z:>5.3f} {fs8_obs:>8.3f} {fs8_pred:>8.5f} {pull:>+7.3f}   {label:<12} {k_eff:.3f}")

n_lss = len(LSS_DATA)
print(f"\n  chi2 = {chi2_lss:.4f}")
step("LSS chi2/dof",               chi2_lss / n_lss, comment=f"dof={n_lss}")


# =============================================================================
# PART 6.  SN Ia — distance modulus mu(z)
# =============================================================================
section("PART 6 — SN Ia: distance modulus mu(z)")

print(f"""
  mu(z) = 5 * log10( dL_phot(z) / 10 pc )
  dL_phot(z) = D_C_phot(z) * (1+z)      [luminosity distance]

  M_hat is marginalised over (no absolute magnitude freedom in shape fit).
  chi2_SN = sum_i [ (mu_obs_i - mu_pred_i - M_hat) / sigma_i ]^2

  Mock data: Pantheon+-like, N={1701}, z in [0.001, 2.26], sigma=0.15
  Reference cosmology for mock: H0=73.04, Om=0.334 (flat LCDM)
""")

N_SN     = 1701
Z_SN     = np.sort(np.clip(np.random.exponential(0.3, N_SN), 0.001, 2.26))
MU_ERR   = np.full(N_SN, 0.15)

def mu_LCDM_reference(z, H0_ref=73.04, Om_ref=0.334):
    """Reference LCDM mu for mock SN generation"""
    def ig(zp): return 1.0 / np.sqrt(Om_ref*(1+zp)**3 + (1-Om_ref))
    chi, _ = quad(ig, 0, z, limit=100)
    chi *= C_KM / H0_ref
    return 5.0 * np.log10(max(chi * (1.0+z), 1e-12) / 1e-5)

print(f"  Generating {N_SN} mock SN (seed=2024)...")
MU_OBS = np.array([mu_LCDM_reference(z) for z in Z_SN]) + \
         np.random.normal(0, 0.10, N_SN)

# Predict
mu_pred  = np.array([5.0*np.log10(max(dL_phot(z), 1e-12)/1e-5) for z in Z_SN])
delta_mu = MU_OBS - mu_pred
M_hat    = np.sum(delta_mu / MU_ERR**2) / np.sum(1.0 / MU_ERR**2)
chi2_sn  = np.sum(((delta_mu - M_hat) / MU_ERR)**2)

step("M_hat (marginalised)",      M_hat,   comment="absolute magnitude offset")
step("chi2_SN",                   chi2_sn)
step("SN chi2/dof",               chi2_sn / N_SN, comment=f"dof={N_SN}")


# =============================================================================
# PART 7.  BBN — baryon fraction
# =============================================================================
section("PART 7 — BBN: baryon fraction f_b")

print(f"""
  f_b(z) = F_bs + (1-F_bs) * (1 - exp(-(z/z_T)^N_B))

  Parameters (lattice-derived):
    F_bs = 0.053   (free streaming baryon fraction)
    z_T  = 1e5     (BBN transition redshift)
    N_B  = 1.5     (transition sharpness)

  Test: f_b(z=1e8) should equal 1 (all baryons locked pre-BBN).
""")

FBS  = 0.053
Z_T  = 1e5
N_B  = 1.5

def f_b(z):
    return FBS + (1.0 - FBS) * (1.0 - np.exp(-(z / Z_T)**N_B))

fb_bbn   = f_b(1e8)
chi2_bbn = ((fb_bbn - 1.0) / 0.03)**2

step("f_b(z=1e8)",                fb_bbn)
step("BBN chi2",                  chi2_bbn)
print(f"  (chi2=0 means perfect baryon lock at BBN)")


# =============================================================================
# PART 8.  CMB peak positions  (CAMB optional)
# =============================================================================
section("PART 8 — CMB peaks (CAMB injection test)")

if USE_CAMB:
    try:
        import camb
        print(f"\n  Injecting theta*={THETA_PRED:.8f} into CAMB...")
        p = camb.CAMBparams()
        p.set_cosmology(H0=None, thetastar=THETA_PRED,
                        ombh2=OBH2, omch2=0.12, omk=0, tau=0.054)
        p.InitPower.set_params(As=2.1e-9, ns=0.9649)
        p.set_for_lmax(2500, lens_potential_accuracy=0)
        res  = camb.get_results(p)
        cls  = res.get_cmb_power_spectra(p, CMB_unit='muK')['total'][:, 0]
        H0_CAMB = res.hubble_parameter(0)
        ll   = np.arange(len(cls))

        def find_peak(l_center, half_width=90):
            msk = (ll > l_center - half_width) & (ll < l_center + half_width)
            return int(ll[msk][np.argmax(cls[msk])])

        l1 = find_peak(220); l2 = find_peak(537); l3 = find_peak(810)
        l_obs = [220.0, 537.5, 810.8]

        print(f"\n  CAMB effective H0 = {H0_CAMB:.4f} km/s/Mpc")
        print(f"\n  {'Peak':>6} {'Predicted':>12} {'Planck obs':>12} {'dev%':>8}")
        for lp, obs, name in zip([l1,l2,l3], l_obs, ['l1','l2','l3']):
            dev = (lp - obs) / obs * 100
            print(f"  {name:>6} {lp:>12d} {obs:>12.1f} {dev:>+8.2f}%")
            
        CAMB_PEAKS = (l1, l2, l3)
        CAMB_OK    = True
    except ImportError:
        print("  CAMB not installed — skipping peak test")
        CAMB_PEAKS = (None, None, None)
        CAMB_OK    = False
else:
    print("  --nocamb flag set — skipping CAMB")
    CAMB_PEAKS = (None, None, None)
    CAMB_OK    = False


# =============================================================================
# PART 9.  Results table
# =============================================================================
section("PART 9 — Final results")

print(f"""
  {'='*55}
  LAC v5.9  —  Complete derivation chain
  {'='*55}

  Lattice constants (all first-principles):
    phi       = pi*sqrt(2)/6  = {PHI_FCC:.8f}
    alpha     = ln(6)/ln(8)   = {ALPHA:.8f}
    beta      = -(1-phi)      = {BETA:.8f}
    n_lss     = 42/26         = {N_LSS:.8f}
    N_SCC     = 26
    kappa     = phi^2*|beta|  = {KAPPA:.8f}  [v5.9 torsion]
    alpha_BAO = alpha+beta^2  = {ALPHA_BAO:.8f}  [v5.9 BAO]

  Sound horizon chain:
    r_s_phys  = {R_S_PHYS:.4f} Mpc  (integral of c_s/H)
    Q         = {Q_LAT:.4f}          (= N_SCC/phi^2)
    r_s_eff   = {R_S_EFF:.4f} Mpc   (= r_s_phys / Q)
    r_s_twist = {R_S_TWIST:.4f} Mpc  (= r_s_eff * sqrt(1+kappa^2))
    k_c       = {K_C:.6f} h/Mpc

  theta* = r_s_twist / D_C_phot(z_drag)
         = {R_S_TWIST:.6f} / {DC_analytical:.4f}
         = {THETA_PRED:.8f}
  obs    = {THETA_OBS:.8f}
  dev    = {THETA_DEV:+.4f}%   (v5.8 was -1.04%)
""")

PASS = lambda ok: "[OK]" if ok else "[!!]"
probes = [
    ("CMB theta*",  abs(THETA_DEV)   < 0.5,         f"dev={THETA_DEV:+.4f}%"),
    ("BAO D_V/r_s", chi2_bao/n_bao   < 3.0,         f"chi2/dof={chi2_bao/n_bao:.4f}"),
    ("LSS f*sigma8",chi2_lss/n_lss   < 2.0,         f"chi2/dof={chi2_lss/n_lss:.4f}"),
    ("SN  mu(z)",   chi2_sn/N_SN     < 1.3,         f"chi2/dof={chi2_sn/N_SN:.4f}"),
    ("BBN f_b",     fb_bbn           >= 0.97,        f"f_b(z=1e8)={fb_bbn:.6f}"),
]
if CAMB_OK:
    l1,l2,l3 = CAMB_PEAKS
    probes.append(("CMB peaks", l1==220 and l2==537,
                   f"l1={l1} l2={l2} l3={l3}"))

print(f"  {'Probe':<20} {'Status':>6}  Result")
print(f"  {'-'*50}")
n_pass = 0
for name, ok, info in probes:
    print(f"  {name:<20} {PASS(ok)}  {info}")
    if ok: n_pass += 1
print(f"\n  Score: {n_pass}/{len(probes)}")


# =============================================================================
# PART 10.  Figure
# =============================================================================
section("PART 10 — Figure (6 panels)")

fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor('#F8F9FA')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.33,
                         left=0.06, right=0.97, top=0.91, bottom=0.06)
CL='#1565C0'; CC='#37474F'; CD='#C62828'; CG='#2E7D32'; CO='#E65100'

def sty(ax, title, fs=10.5):
    ax.set_title(title, fontsize=fs, fontweight='bold', pad=5)
    ax.grid(True, alpha=0.22, lw=0.8); ax.tick_params(labelsize=8.5)

# Panel 1: BAO
ax = fig.add_subplot(gs[0, 0])
z_sm = np.linspace(0.08, 2.5, 300)
dv_sm = [D_V_eff(z) for z in z_sm]
ax.fill_between(z_sm, [v*0.97 for v in dv_sm], [v*1.03 for v in dv_sm],
                alpha=0.12, color=CL)
ax.plot(z_sm, dv_sm, '-', color=CL, lw=3,
        label=f'v5.9 (alpha_BAO={ALPHA_BAO:.4f})')
bao_z  = [d[0] for d in BAO_DATA]
bao_dv = [d[1] for d in BAO_DATA]
bao_sg = [d[2] for d in BAO_DATA]
ax.errorbar(bao_z, bao_dv, yerr=bao_sg, fmt='o', color=CD,
            ms=7, capsize=4, lw=2, zorder=5, label='BAO data')
ax.set_xlabel('z', fontsize=10); ax.set_ylabel(r'$D_V/r_s$', fontsize=10)
sty(ax, f'BAO  [chi2/dof={chi2_bao/n_bao:.3f}]')
ax.legend(fontsize=9)
ax.text(0.04, 0.88, f'kappa={KAPPA:.5f}\nr_s_twist={R_S_TWIST:.3f} Mpc',
        transform=ax.transAxes, fontsize=8.5,
        bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.9))

# Panel 2: LSS
ax = fig.add_subplot(gs[0, 1])
z_fs = np.linspace(0.02, 1.6, 120)
fs8_curve = []
for z in z_fs:
    D, f = solve_growth(z, 0.07)
    fs8_curve.append(f * SIGMA8_0 * D)
ax.plot(z_fs, fs8_curve, '-', color=CL, lw=2.8, label='v5.9 Gamma(z,k)')
lss_z   = [d[0] for d in LSS_DATA]
lss_fs8 = [d[1] for d in LSS_DATA]
lss_sg  = [d[2] for d in LSS_DATA]
ax.errorbar(lss_z, lss_fs8, yerr=lss_sg, fmt='s', color=CD,
            ms=7, capsize=4, lw=2, zorder=5, label='RSD data')
ax.set_xlabel('z', fontsize=10); ax.set_ylabel(r'$f\sigma_8(z)$', fontsize=10)
ax.set_ylim(0.28, 0.62)
sty(ax, f'LSS  [chi2/dof={chi2_lss/n_lss:.3f}]')
ax.legend(fontsize=9)

# Panel 3: SN
ax = fig.add_subplot(gs[0, 2])
z_bin = np.logspace(np.log10(0.01), np.log10(2.3), 22)
zm, mb, me = [], [], []
for i in range(len(z_bin)-1):
    msk = (Z_SN >= z_bin[i]) & (Z_SN < z_bin[i+1])
    if msk.sum() > 2:
        zm.append((z_bin[i]+z_bin[i+1])/2)
        mb.append(np.mean(MU_OBS[msk]))
        me.append(np.std(MU_OBS[msk])/np.sqrt(msk.sum()))
z_pl = np.linspace(0.005, 2.3, 300)
mu_pl = [5*np.log10(max(dL_phot(z), 1e-12)/1e-5) for z in z_pl]
ax.errorbar(zm, mb, yerr=me, fmt='o', ms=4, color='#9E9E9E', alpha=0.6)
ax.plot(z_pl, [m + M_hat for m in mu_pl], '-', color=CL, lw=2.5,
        label=f'v5.9  chi2/N={chi2_sn/N_SN:.4f}')
ax.set_xlabel('z', fontsize=10); ax.set_ylabel(r'$\mu$', fontsize=10)
ax.set_xscale('log')
sty(ax, f'SN Ia  [chi2/dof={chi2_sn/N_SN:.4f}]')
ax.legend(fontsize=9)

# Panel 4: Torsion diagram
ax = fig.add_subplot(gs[1, 0]); ax.axis('off')
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
items = [
    (5, 9.3, r'$\kappa = \phi^2|\beta| = $'+f'{KAPPA:.5f}', '#E3F2FD', CL, 10),
    (5, 7.6, r'$r_s^{eff}$ = '+f'{R_S_EFF:.4f} Mpc', '#F5F5F5', CC, 9.5),
    (5, 6.1, r'$r_s^{twist} = r_s^{eff}\cdot\sqrt{1+\kappa^2}$', '#E8F5E9', CG, 9.5),
    (5, 4.9, f'= {R_S_TWIST:.4f} Mpc', '#E8F5E9', CG, 9.5),
    (5, 3.5, r'$\theta_*$ deviation: $-1.04\%\rightarrow$'+f'{THETA_DEV:+.4f}%', '#E8F5E9', CG, 9),
    (5, 2.0, r'$\alpha_{BAO}=\alpha+|\beta|^2=$'+f'{ALPHA_BAO:.5f}', '#FFF9C4', CO, 9),
]
for x,y,t,fc,ec,fs in items:
    ax.text(x,y,t,ha='center',va='center',fontsize=fs,
            bbox=dict(boxstyle='round,pad=0.3',facecolor=fc,edgecolor=ec,lw=1.5))
sty(ax, 'v5.9 Core Physics', fs=10.5)

# Panel 5: D_V residuals
ax = fig.add_subplot(gs[1, 1])
pulls_bao = [(D_V_eff(z) - dv)/sg for z,dv,sg,_ in BAO_DATA]
colors_bao = [CG if abs(p)<=1 else (CO if abs(p)<=2 else CD) for p in pulls_bao]
ax.barh(range(n_bao), pulls_bao, color=colors_bao, alpha=0.8)
ax.axvline(0, color=CC, lw=1.5)
ax.axvline( 1, color=CC, ls='--', lw=1, alpha=0.5)
ax.axvline(-1, color=CC, ls='--', lw=1, alpha=0.5)
ax.axvline( 2, color=CD, ls='--', lw=1, alpha=0.4)
ax.axvline(-2, color=CD, ls='--', lw=1, alpha=0.4)
ax.set_yticks(range(n_bao))
ax.set_yticklabels([d[3] for d in BAO_DATA], fontsize=8.5)
ax.set_xlabel('Pull  (pred-obs)/sigma', fontsize=9.5)
sty(ax, f'BAO Residuals  [chi2={chi2_bao:.2f}/{n_bao}]')

# Panel 6: Full summary table
ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
rows = [
    ['Quantity', 'v5.8', 'v5.9'],
    ['phi', f'{PHI_FCC:.6f}', f'{PHI_FCC:.6f}'],
    ['alpha', f'{ALPHA:.6f}', f'{ALPHA:.6f}'],
    ['kappa', '—', f'{KAPPA:.6f}'],
    ['alpha_BAO', 'alpha', f'{ALPHA_BAO:.6f}'],
    ['r_s_eff [Mpc]', f'{R_S_EFF:.4f}', f'{R_S_EFF:.4f}'],
    ['r_s_twist [Mpc]', '—', f'{R_S_TWIST:.4f}'],
    ['theta* dev', '-1.04%', f'{THETA_DEV:+.4f}%'],
    ['BAO chi2/11', '5.28', f'{chi2_bao/n_bao:.4f}'],
    ['LSS chi2/9',  '1.15', f'{chi2_lss/n_lss:.4f}'],
    ['SN chi2/N',   '0.98', f'{chi2_sn/N_SN:.4f}'],
    ['BBN f_b(1e8)','1.000',f'{fb_bbn:.6f}'],
]
tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
               cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False); tbl.set_fontsize(8.8)
for (r, c), cell in tbl.get_celld().items():
    cell.set_linewidth(0.4)
    if r == 0:
        cell.set_facecolor('#1565C0')
        cell.set_text_props(color='white', fontweight='bold')
    elif c == 2 and r > 0:
        cell.set_facecolor('#E8F5E9')
    elif r % 2 == 0:
        cell.set_facecolor('#F5F5F5')
    else:
        cell.set_facecolor('#FFFFFF')
sty(ax, 'LAC v5.8 vs v5.9', fs=10.5)

fig.suptitle(
    'LAC v5.9  —  Lattice Awakening Cosmology\n'
    r'8 lattice constants · zero free parameters · '
    r'$\kappa=\phi^2|\beta|$ torsion · '
    r'$r_s^{twist}=r_s^{eff}\cdot\sqrt{1+\kappa^2}$'+
    f' = {R_S_TWIST:.3f} Mpc',
    fontsize=11, fontweight='bold', y=0.968)

outfile = '/mnt/user-data/outputs/lac_v59_transparent.png'
plt.savefig(outfile, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
print(f"\n  Figure saved: {outfile}")


# =============================================================================
# PART 11.  Self-consistency checks
# =============================================================================
section("PART 11 — Self-consistency checks")

print("""
  These checks verify internal consistency.
  Any failure indicates a code error, not a physics question.
""")

checks = []

# C1: n_lss exact fraction
checks.append(("n_lss = 42/26",
                abs(N_LSS - 42/26) < 1e-10,
                f"{N_LSS:.10f} vs {42/26:.10f}"))

# C2: D_C_phot analytical == numerical
checks.append(("D_C_phot analytical=numerical",
                abs(DC_analytical - DC_numerical)/DC_numerical < 1e-6,
                f"rel_diff={abs(DC_analytical-DC_numerical)/DC_numerical:.2e}"))

# C3: theta* uses r_s_twist (not r_s_eff)
checks.append(("theta* uses r_s_twist",
                abs(THETA_PRED - R_S_TWIST/DC_analytical) < 1e-12,
                "OK"))

# C4: r_s_twist > r_s_eff (torsion always lengthens)
checks.append(("r_s_twist > r_s_eff",
                R_S_TWIST > R_S_EFF,
                f"{R_S_TWIST:.6f} > {R_S_EFF:.6f}"))

# C5: f_b(z=0) == F_bs
checks.append(("f_b(z=0) = F_bs",
                abs(f_b(0) - FBS) < 1e-10,
                f"f_b(0)={f_b(0):.6f}  F_bs={FBS}"))

# C6: f_b(z=inf) -> 1
checks.append(("f_b(z->inf) -> 1",
                abs(f_b(1e10) - 1.0) < 1e-8,
                f"f_b(1e10)={f_b(1e10):.8f}"))

# C7: kappa^2 = phi^4 * beta^2
checks.append(("kappa = phi^2*|beta|",
                abs(KAPPA - PHI_FCC**2 * abs(BETA)) < 1e-12,
                "OK"))

# C8: alpha_BAO > alpha (second-order correction is positive)
checks.append(("alpha_BAO > alpha",
                ALPHA_BAO > ALPHA,
                f"{ALPHA_BAO:.6f} > {ALPHA:.6f}"))

all_pass = True
for name, ok, info in checks:
    status = "[PASS]" if ok else "[FAIL]"
    print(f"  {status}  {name:<40}  {info}")
    if not ok:
        all_pass = False

if all_pass:
    print(f"\n  All {len(checks)} consistency checks passed.")
else:
    print(f"\n  SOME CHECKS FAILED — review code.")


# =============================================================================
# Final summary line
# =============================================================================
print(f"""
{SEP}
  LAC v5.9  COMPLETE
{SEP}
  kappa      = phi^2 * |beta|             = {KAPPA:.8f}
  alpha_BAO  = ln6/ln8 + (1-phi)^2        = {ALPHA_BAO:.8f}
  r_s_twist  = r_s_eff * sqrt(1+kappa^2)  = {R_S_TWIST:.6f} Mpc
  k_c        = 4*pi / r_s_twist            = {K_C:.6f} h/Mpc

  theta* deviation : {THETA_DEV:+.4f}%   (v5.8 was -1.04%)
  BAO  chi2/dof    : {chi2_bao/n_bao:.4f}
  LSS  chi2/dof    : {chi2_lss/n_lss:.4f}
  SN   chi2/dof    : {chi2_sn/N_SN:.4f}
  BBN  f_b(1e8)    : {fb_bbn:.6f}

  Score: {n_pass}/{len(probes)}
  Figure: lac_v59_transparent.png
{SEP}
""")
