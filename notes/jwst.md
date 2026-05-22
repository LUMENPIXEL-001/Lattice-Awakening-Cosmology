# Internal Note: JWST-Extension Exploration of v1.3 Ladder

**LUMENPIXEL, May 2026**
**Status: Internal working note. Not a published paper.**
**Purpose: Record exploratory result for future development.**

---

## Summary

The v1.3 same-sign damped ladder (T(z) = 1 + Σ_n A_n sech²((θ − n ln φ)/w), 
A_n = A_1/φ^(n−1), w = (ln φ)³, A_1 = (ln φ)⁴) admits a natural extension to 
n_max = 5, 6 that reaches the JWST early-galaxy regime (z ≈ 10–17) without 
introducing new free parameters beyond v1.3.

Key fact: in z-coordinate, rung centres lie at z_n = φ^n − 1:
- n=5: z = 10.09  (JWST regime entry)
- n=6: z = 16.94  (JWST regime depth)

This is a direct consequence of the θ = n ln φ ladder, not an imposed match.

---

## Quantitative results (computed via lvc_v13 reproduction pipeline)

**T(z) − 1 at JWST redshifts:**

| z    | n_max=1   | n_max=3   | n_max=5   | n_max=6   |
|------|-----------|-----------|-----------|-----------|
| 10.0 | 2.2×10⁻¹⁶ | 3.0×10⁻⁹  | 7.8×10⁻³  | 7.8×10⁻³  |
| 13.0 | 0         | 3.9×10⁻¹¹ | 4.6×10⁻⁴  | 6.8×10⁻⁴  |
| 17.0 | 0         | 4.3×10⁻¹³ | 5.3×10⁻⁶  | 4.8×10⁻³  |

**Growth ratio D_LVC/D_LCDM at JWST redshifts (α = 0.2258, n_max-matched):**

With α refit per n_max to recover σ₈ = 0.811:

| n_max | α*     | growth boost at z=10 | at z=13 | at z=17 |
|-------|--------|---------------------|---------|---------|
| 1     | 0.2258 | ~0                  | ~0      | ~0      |
| 5     | 0.0483 | +0.5%               | ~0      | ~0      |
| 6     | 0.0402 | +4.9%               | +3.0%   | +0.5%   |

**BAO chi² (PP-excluded, N=37):**

| n_max | χ²      | Δχ² vs v1.1 n=1 |
|-------|---------|-----------------|
| 1     | 56.13   | 0               |
| 2     | 53.72   | −2.42           |
| 3     | 53.28   | −2.85           |
| 5     | 53.31   | −2.83           |
| 6     | 53.31   | −2.83           |

Extending n_max from 3 to 6 does not damage late-time BAO fit 
(Δχ² ≈ +0.03 across n_max = 3 → 6).

---

## What is preserved from v1.3

- v1.1 sine-Gordon Lagrangian (single field, unchanged)
- v1.3 non-minimal coupling f(ϑ) = exp[−α(1 − cos(ϑ/2))]  (unchanged form)
- v1.3 same-sign damped amplitude pattern A_n = A_1/φ^(n−1) (unchanged)
- v1.1 width w = (ln φ)³ (unchanged)
- Bounded-Closure axiom z_c = 1/φ (unchanged)
- All natural-constant locks (no new constants introduced)

## What changes

- n_max: 1 (v1.1) or 2 (v1.3 marginal) → 6 (JWST extension)
- α value: 0.226 → 0.040 (re-tuned for σ₈ = 0.811)
- Free parameter count: unchanged (still α as the only continuous freedom)

## Status of the extension

This extension is NOT required by current data:
- BAO + Pantheon+ data prefer n_max = 2 marginally (1.48σ, v1.3 §3.3)
- Current data does not require n_max > 2 (v1.3 §6, "higher-rung saturation")

This extension MAY be required by future data:
- If JWST early-galaxy tension is established as a cosmological (not astrophysical) 
  problem, this extension provides the natural LVC response
- DR3 BAO and future high-z probes will determine which n_max is preferred

Position of this note in LVC programme:
- Not a published candidate
- Recorded as exploratory result confirming that JWST scenario can be absorbed
  within the existing Lagrangian family without structural complication
- Consistent with "scope-restricted path" of v1.4 S1 statement; the J1 (unified 
  Lagrangian) member remains open for future development

---

## Reproduction

All numbers above produced from lvc_v13_reproduce.py with T_ladder extended 
to n_max ∈ {5, 6}. The relevant function modifications:

```python
def T_ladder_extended(z, n_max=6):
    theta = np.log(1.0 + z)
    bump = np.zeros_like(theta)
    for n in range(1, n_max + 1):
        amp = LNPHI_4 / (PHI ** (n - 1))
        u = np.clip((theta - n * LNPHI) / LNPHI_3, -50, 50)
        bump += amp / np.cosh(u) ** 2
    return 1.0 + bump
```

Same structure for the kink phase used in G_eff_natural.

---

## Limitations

1. The 4.9% growth boost at z = 10 is at the lower end of what some JWST 
   interpretations require (5-10%, with aggressive readings up to 30%). 
   Whether this is sufficient depends on which JWST interpretation hardens.

2. v1.3 §6 explicitly notes that "the data do not require n_max > 2." 
   Extending to n_max = 6 is a forward-looking provision, not a current 
   data requirement.

3. Full Boltzmann CMB likelihood not evaluated for extended ladder; only 
   distance-prior-level compatibility checked through v1.3 pipeline.

4. The amplitude pattern A_n = A_1/φ^(n−1) is preserved by choice; whether 
   future high-z data prefers a different pattern at n ≥ 5 is open.

---

## Hook for future development

If JWST tension hardens as cosmological:
- Promote this note to formal extension paper
- Test n_max = 7, 8 reaching z ~ 28, 46 (21cm regime)
- Compare with then-current data on early structure formation

If JWST tension dissolves as astrophysical:
- Retain this note as record that LVC could have absorbed the scenario
- No publication required

---

**End of internal note.**
