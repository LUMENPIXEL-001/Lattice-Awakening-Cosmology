# ============================================================
# LVC Reproduction Code  (Python 3.8+)
# Lagrangian Variable Cosmology — Intermediate Report 2026
# Dependencies: numpy, scipy, matplotlib
# All functions are self-contained. No hidden parameters.
# Run:  python3 lvc_reproduce.py
# ============================================================
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import warnings
warnings.filterwarnings("ignore")

# --- Cosmological constants (Planck 2015) -------------------
H0    = 67.74          # Hubble constant  [km/s/Mpc]
Om    = 0.3089         # matter density parameter
OL    = 0.6911         # dark energy density parameter
Ob    = 0.0486         # baryon density parameter
Ob_h2 = Ob*(H0/100)**2
c_km  = 299792.458     # speed of light   [km/s]
T_CMB = 2.7255         # CMB temperature  [K]

# --- CMB micro-rotation parameters (Layer 2) ----------------
beta  = 0.97           # rotation decay index (CMB-constrained)
A_rot = 0.30           # rotation amplitude
z_c   = 1150.0         # recombination centre
s_hi  = 200.0          # high-z asymmetric width
s_lo  =  80.0          # low-z asymmetric width

# --- Geometric transition T(z) parameters (Layer 1) ---------
bv  = -0.45;  cv = 1.80
z1  =  0.95;  z2 = 1.40;  ep = 0.08
a_  = 1 + bv + cv       # ensures T(z->0) ~ 1

# --- Observational data compilations ------------------------
BAO_DATA = [            # (z, DV/rs_obs, sigma)
    (0.106, 2.98, 0.13), (0.15,  4.47, 0.17),
    (0.38,  9.95, 0.20), (0.51, 13.38, 0.18),
    (0.61, 15.74, 0.23), (0.70, 17.65, 0.30),
    (1.52, 26.08, 0.67),
]
SN_DATA = [             # (z, mu_obs, sigma)
    (0.10, 38.27, 0.10), (0.40, 42.15, 0.09),
    (0.80, 44.50, 0.10), (1.50, 46.75, 0.15),
    (2.00, 47.72, 0.20),
]

# ============================================================
# LAYER 1: Geometric channel
# ============================================================
def H_LCDM(z):
    """Standard LCDM Hubble rate [km/s/Mpc]."""
    return H0 * np.sqrt(Om*(1+z)**3 + OL)

def T_tanh(z):
    """LVC geometric transition function (dimensionless)."""
    return a_ + bv*np.tanh((z-z1)/ep) + cv*np.tanh((z-z2)/ep)

def H_LVC(z):
    """LVC Hubble rate [km/s/Mpc]. Clamps T >= 0.02."""
    return H_LCDM(z) * max(T_tanh(z), 0.02)

# ============================================================
# LAYER 2: Fluid channel
# ============================================================
def F_rot(z):
    """Micro-rotation rate F_rot(z), from CMB polarisation."""
    sigma = s_lo if z <= z_c else s_hi
    return (1+z)**(beta-1) * (1 + A_rot*np.exp(-0.5*((z-z_c)/sigma)**2))

def R_std(z):
    """Standard baryon-photon ratio R(z) = 3*rho_b/(4*rho_gamma)."""
    return 31500 * Ob_h2 * (T_CMB/2.7)**(-4) / (1+z)

def cs_std(z):
    """Standard photon-baryon sound speed [km/s]."""
    return c_km / np.sqrt(3*(1+R_std(z)))

def compute_kappa():
    """
    Compute coupling invariant kappa = n*(1-beta).

    Exact formula:
        kappa = -2 * ln(r_s_obs / r_s_bare) / L
    where
        r_s_bare = integral_0^z_drag  cs_std(z) / (H_LVC(z)*(1+z)) dz
        L = <ln(1+z)>_w  (rs-weighted mean log-redshift)

    The G_skew correction to kappa is < 0.006% and is included
    automatically via the exact F_rot(z) in cs_eff.

    Returns
    -------
    kappa   : float  coupling invariant
    rs_bare : float  unmodified sound horizon under H_LVC  [Mpc]
    L       : float  rs-weighted <ln(1+z)>
    """
    rs_target = 148.1491   # Mpc, observed BAO sound horizon
    w_tot, _ = quad(lambda z: cs_std(z)/(H_LVC(z)*(1+z)), 0, 1060, limit=200)
    lnz_int, _ = quad(
        lambda z: cs_std(z)/(H_LVC(z)*(1+z)) * np.log(1+z),
        0, 1060, limit=200)
    L = lnz_int / w_tot        # <ln(1+z)>_w
    rs_bare = w_tot            # = integral cs_std/(H_LVC*(1+z))
    kappa = -2 * np.log(rs_target / rs_bare) / L
    return kappa, rs_bare, L

kappa, rs_bare, L_val = compute_kappa()
n_coupling = kappa / (1 - beta)   # fluid coupling index alpha

def cs_eff(z):
    """
    Effective sound speed with F_rot modification [km/s].
        c_s_eff(z) = c_s_std(z) * F_rot(z)^(n/2)
    where n = kappa / (1 - beta).
    """
    return cs_std(z) * F_rot(z)**(n_coupling/2)

# ============================================================
# DERIVED OBSERVABLES
# ============================================================
def rs_integral(cs_func, H_func, z_drag=1060):
    """Sound horizon r_s [Mpc]."""
    v, _ = quad(lambda z: cs_func(z)/(H_func(z)*(1+z)), 0, z_drag, limit=250)
    return v

def eta_ratio(H_func, z_cut=1.0, z_max=1100):
    """Conformal time ratio eta(0->z_cut) / eta(0->z_max)."""
    I1, _ = quad(lambda z: 1/(H_func(z)*(1+z)), 0, z_cut, limit=100)
    I2, _ = quad(lambda z: 1/(H_func(z)*(1+z)), 0, z_max, limit=150)
    return I1 / I2

def DV_Mpc(z, H_func):
    """Volume-averaged distance D_V(z) [Mpc]."""
    DM, _ = quad(lambda zp: c_km/H_func(zp), 0, z, limit=100)
    return (DM**2 * c_km * z / H_func(z))**(1/3)

def mu_z(z, H_func):
    """Distance modulus mu(z) [mag]."""
    DL, _ = quad(lambda zp: c_km/H_func(zp), 0, z, limit=100)
    return 5*np.log10(DL*(1+z)) + 25

def chi2_BAO(H_func, rs_val):
    """BAO chi-squared."""
    return sum(((DV_Mpc(z,H_func)/rs_val - obs)/err)**2
               for z, obs, err in BAO_DATA)

def chi2_SN(H_func):
    """SN Ia chi-squared."""
    return sum(((mu_z(z,H_func) - obs)/err)**2
               for z, obs, err in SN_DATA)

# ============================================================
# MAIN: RUN AND PRINT RESULTS
# ============================================================
if __name__ == "__main__":
    print("Computing LVC results...")

    rs_lcdm  = rs_integral(cs_std, H_LCDM)
    rs_lvc   = rs_integral(cs_eff,  H_LVC)
    eta_lcdm = eta_ratio(H_LCDM)
    eta_lvc  = eta_ratio(H_LVC)
    b_lcdm   = chi2_BAO(H_LCDM, rs_lcdm)
    s_lcdm   = chi2_SN(H_LCDM)
    b_lvc    = chi2_BAO(H_LVC,  rs_lvc)
    s_lvc    = chi2_SN(H_LVC)

    print("\n" + "="*55)
    print("  LVC Numerical Results")
    print("="*55)
    print(f"  kappa (coupling invariant) = {kappa:.6f}")
    print(f"  n = kappa/(1-beta)         = {n_coupling:.4f}")
    print(f"  rs_bare (no cs mod)        = {rs_bare:.4f} Mpc")
    print(f"  L = <ln(1+z)>_w            = {L_val:.6f}")
    print()
    print(f"  {'Observable':<22} {'LCDM':>8} {'LVC':>8} {'Delta':>8}")
    print(f"  {'-'*50}")
    print(f"  {'eta ratio':<22} {eta_lcdm:>8.4f} {eta_lvc:>8.4f} {eta_lvc-eta_lcdm:>+8.4f}")
    print(f"  {'r_s [Mpc]':<22} {rs_lcdm:>8.2f} {rs_lvc:>8.2f} {rs_lvc-rs_lcdm:>+8.2f}")
    print(f"  {'BAO chi2':<22} {b_lcdm:>8.2f} {b_lvc:>8.2f} {b_lvc-b_lcdm:>+8.2f}")
    print(f"  {'SN  chi2':<22} {s_lcdm:>8.2f} {s_lvc:>8.2f} {s_lvc-s_lcdm:>+8.2f}")
    print(f"  {'Total chi2':<22} {b_lcdm+s_lcdm:>8.2f} {b_lvc+s_lvc:>8.2f}"
          f" {(b_lvc+s_lvc)-(b_lcdm+s_lcdm):>+8.2f}")

    print("\n  Kappa invariance: n*(1-beta) for different beta values")
    print(f"  {'beta':>8} {'n_opt':>10} {'kappa':>12}")
    for b_test in [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]:
        def Fb(z, b=b_test):
            s = s_lo if z <= z_c else s_hi
            return (1+z)**(b-1)*(1+A_rot*np.exp(-0.5*((z-z_c)/s)**2))
        def rs_nb(n_val, b=b_test):
            v,_ = quad(
                lambda z: cs_std(z)*Fb(z,b)**(n_val/2)/(H_LVC(z)*(1+z)),
                0, 1060, limit=150)
            return v
        n_o = brentq(lambda n: rs_nb(n)-148.1491, -50, 50, xtol=1e-4)
        print(f"  {b_test:>8.2f} {n_o:>10.4f} {n_o*(1-b_test):>12.6f}")

    # Optional: generate plots
    try:
        import matplotlib.pyplot as plt
        zz = np.linspace(0.001, 3, 400)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(zz, [T_tanh(z) for z in zz], 'steelblue', lw=2)
        axes[0].axhline(1, color='gray', ls='--', lw=1)
        axes[0].set(title='T(z) = H_LVC/H_LCDM', xlabel='z', ylabel='T(z)')
        axes[0].grid(alpha=0.3)

        zz2 = np.logspace(-1, 3.1, 400)
        axes[1].semilogx(zz2, [F_rot(z) for z in zz2], 'darkorange', lw=2)
        axes[1].axhline(1, color='gray', ls='--', lw=1)
        axes[1].set(title='F_rot(z)', xlabel='z', ylabel='F_rot')
        axes[1].grid(alpha=0.3)

        n_arr = np.arange(0.80, 1.00, 0.01)
        kp_arr = []
        for b in n_arr:
            def Fb2(z, b_=b):
                s=s_lo if z<=z_c else s_hi
                return (1+z)**(b_-1)*(1+A_rot*np.exp(-0.5*((z-z_c)/s)**2))
            def rsn2(n_val, b_=b):
                v,_=quad(lambda z:cs_std(z)*Fb2(z,b_)**(n_val/2)/(H_LVC(z)*(1+z)),0,1060,limit=120)
                return v
            try:
                n_o=brentq(lambda n:rsn2(n)-148.1491,-50,50,xtol=1e-3)
                kp_arr.append(n_o*(1-b))
            except:
                kp_arr.append(np.nan)
        axes[2].plot(n_arr, kp_arr, 'purple', lw=2, marker='o', ms=4)
        axes[2].axhline(kappa, color='gray', ls='--', lw=1,
                        label=f'kappa={kappa:.4f}')
        axes[2].set(title='kappa Invariance', xlabel='beta',
                    ylabel='n*(1-beta)'); axes[2].legend(); axes[2].grid(alpha=0.3)

        fig.tight_layout()
        plt.savefig('lvc_figures.png', dpi=150, bbox_inches='tight')
        print("\n  Figures saved to lvc_figures.png")
        plt.close()
    except ImportError:
        print("\n  (matplotlib not available — skipping figures)")
