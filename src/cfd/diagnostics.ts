/**
 * Diagnostic definitions shared by the MAC reference, tests, and any UI readout
 * that claims a physical quantity. Preview skills that still use the collocated
 * field must not reuse these names for a different stencil.
 *
 * Kinetic energy uses the staggered face velocities, averaged per cell:
 *   ke = Σ ½ ρ ((u_w²+u_e²)/2 + (v_s²+v_n²)/2 + (w_b²+w_f²)/2) ΔV
 * Viscous dissipation (power, Watts) is the incompressible strain contraction
 *   P = Σ 2 μ S:S ΔV,   μ = ρ ν,   S = (∇u + ∇uᵀ)/2
 * so that, with no boundary work, d(ke)/dt = −P.
 */

export type StrainSample = {
  sxx: number;
  syy: number;
  szz: number;
  sxy: number;
  sxz: number;
  syz: number;
};

export function strainContraction(s: StrainSample): number {
  return s.sxx * s.sxx + s.syy * s.syy + s.szz * s.szz + 2 * (s.sxy * s.sxy + s.sxz * s.sxz + s.syz * s.syz);
}

export function dissipationDensity(nu: number, s: StrainSample): number {
  return 2 * nu * strainContraction(s);
}
