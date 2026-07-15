"""
PhysicsResiduals — discrete PDE residuals for all physics constraints.

State tensor convention
-----------------------
s : (B, N, 7)   where channels are:
  0  u850   zonal wind at 850 hPa         [m/s]
  1  v850   meridional wind at 850 hPa    [m/s]
  2  u500   zonal wind at 500 hPa         [m/s]
  3  v500   meridional wind at 500 hPa    [m/s]
  4  z500   geopotential height (anom.)   [m]
  5  T2m    2-m temperature (anom.)       [K]
  6  MSLP   mean sea-level pressure (anom.) [hPa]

Physics implemented
-------------------

1. Momentum / Advection  (primitive equation, 850 hPa layer)
   ─────────────────────────────────────────────────────────
   ∂u/∂t + u ∂u/∂x + v ∂u/∂y + (1/ρ) ∂p/∂x − fv − ν∇²u = 0   (1a)
   ∂v/∂t + u ∂v/∂x + v ∂v/∂y + (1/ρ) ∂p/∂y + fu − ν∇²v = 0   (1b)

   Time derivative approximated by Euler forward:
     ∂u/∂t ≈ (u_{t+1} − u_t) / Δt

   All spatial derivatives are computed with GraphDifferentialOps.

2. Temperature Advection-Diffusion
   ─────────────────────────────────────────────────────────
   ∂T/∂t + u ∂T/∂x + v ∂T/∂y − κ ∇²T = 0                      (2)

3. Mass Conservation (Incompressibility)
   ─────────────────────────────────────────────────────────
   ∇·u = ∂u/∂x + ∂v/∂y = 0                                       (3)

   This is the Boussinesq / large-scale incompressibility approximation.

4. Gradient Wind Balance (Wind-Pressure Consistency)
   ─────────────────────────────────────────────────────────
   V²/r + f V + (1/ρ) ∂p/∂r = 0                                   (4)

   where V = √(u²+v²), r = distance from storm centre, and ∂p/∂r is
   the radial pressure gradient in the wind direction.

   Derivation: In the storm frame, the equation of motion in the radial
   direction (centripetal acceleration = pressure gradient force + Coriolis):
     −V²/r (centripetal) = −(1/ρ) ∂p/∂r − f V
   ⟹  V²/r + f V + (1/ρ) ∂p/∂r = 0

5. Temporal Continuity (Soft Constraint)
   ─────────────────────────────────────────────────────────
   ‖s_{t+1} − s_t‖² / N ≤ (C·Δt)²/C_channels

   where C = 50 m/s is a characteristic velocity scale.
   Penalises unrealistically large state jumps.

6. Kinetic-Energy Conservation (inviscid approx.)
   ─────────────────────────────────────────────────────────
   ∂E/∂t + ∂(Eu)/∂x + ∂(Ev)/∂y = 0,   E = ½(u² + v²)          (6)

All residuals return non-negative mean-squared scalars (dimensionless
after squaring and averaging).  The loss module applies physical unit
weights via the lambda_ hyper-parameters.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from .config import PIGNOConfig
from .graph_builder import GraphDifferentialOps


class PhysicsResiduals:
    """
    Computes the six PDE residuals from consecutive state snapshots.

    This is a plain Python class (not nn.Module) — it has no learnable
    parameters.  It is used by PhysicsInformedLoss as a computation helper.
    """

    def __init__(self, cfg: PIGNOConfig, ops: GraphDifferentialOps):
        self.cfg  = cfg
        self.ops  = ops
        self.f    = cfg.f_coriolis         # s⁻¹
        self.nu   = cfg.nu_viscosity       # m² s⁻¹
        self.kap  = cfg.kappa_diffusivity  # m² s⁻¹
        self.rho  = cfg.rho_air            # kg m⁻³
        self.dt   = cfg.dt_physics         # s

    # ── 1. Advection (momentum equation) ──────────────────────────────────────

    def advection(self, s_t: Tensor, s_tp1: Tensor) -> Tensor:
        """
        Residuals for both wind components at 850 hPa.

        R_u = (u_{t+1}−u_t)/Δt + u_t ∂u_t/∂x + v_t ∂u_t/∂y
              + (1/ρ) ∂p_t/∂x − f v_t − ν ∇²u_t

        R_v = (v_{t+1}−v_t)/Δt + u_t ∂v_t/∂x + v_t ∂v_t/∂y
              + (1/ρ) ∂p_t/∂y + f u_t − ν ∇²v_t

        Returns: mean of MSE(R_u) and MSE(R_v).
        """
        u   = s_t[..., 0:1];   v   = s_t[..., 1:2]
        p   = s_t[..., 6:7]                          # MSLP as pressure proxy
        u1  = s_tp1[..., 0:1]; v1  = s_tp1[..., 1:2]

        du_dx, du_dy = self.ops.gradient(u)
        dv_dx, dv_dy = self.ops.gradient(v)
        dp_dx, dp_dy = self.ops.gradient(p)
        lap_u        = self.ops.laplacian(u)
        lap_v        = self.ops.laplacian(v)

        R_u = ((u1 - u) / self.dt
               + u * du_dx + v * du_dy
               + (1.0 / self.rho) * dp_dx
               - self.f * v
               - self.nu * lap_u)

        R_v = ((v1 - v) / self.dt
               + u * dv_dx + v * dv_dy
               + (1.0 / self.rho) * dp_dy
               + self.f * u
               - self.nu * lap_v)

        return 0.5 * (R_u.pow(2).mean() + R_v.pow(2).mean())

    # ── 2. Temperature diffusion ───────────────────────────────────────────────

    def diffusion(self, s_t: Tensor, s_tp1: Tensor) -> Tensor:
        """
        R_T = (T_{t+1}−T_t)/Δt + u_t ∂T_t/∂x + v_t ∂T_t/∂y − κ ∇²T_t

        Returns: MSE(R_T).
        """
        u   = s_t[..., 0:1];   v   = s_t[..., 1:2]
        T   = s_t[..., 5:6]
        T1  = s_tp1[..., 5:6]

        dT_dx, dT_dy = self.ops.gradient(T)
        lap_T        = self.ops.laplacian(T)

        R_T = ((T1 - T) / self.dt
               + u * dT_dx + v * dT_dy
               - self.kap * lap_T)

        return R_T.pow(2).mean()

    # ── 3. Mass conservation ───────────────────────────────────────────────────

    def mass_conservation(self, s_t: Tensor, s_tp1: Tensor) -> Tensor:
        """
        Incompressibility: the PREDICTED next state must have ∇·u = 0.
        Evaluating on s_tp1 (model prediction) means this residual
        changes as model weights update, providing a real training signal.

        Returns: MSE(∇·u_predicted).
        """
        u = s_tp1[..., 0:1]   # predicted zonal wind
        v = s_tp1[..., 1:2]   # predicted meridional wind
        div_uv = self.ops.divergence(u, v)
        return div_uv.pow(2).mean()

    # ── 4. Wind-pressure (gradient wind balance) ───────────────────────────────

    def wind_pressure(self, s_t: Tensor, s_tp1: Tensor) -> Tensor:
        """
        Gradient wind balance on the PREDICTED next state:
          V²/r + f V + (1/ρ) ∂p/∂r = 0
        Evaluating on s_tp1 provides a training signal that changes
        as the model learns to predict physically consistent wind-pressure.

        Returns: MSE(R_wp on predicted state).
        """
        u  = s_tp1[..., 0:1];  v  = s_tp1[..., 1:2]   # predicted winds
        p  = s_tp1[..., 6:7]                            # predicted MSLP

        V  = (u.pow(2) + v.pow(2)).sqrt() + 1e-6   # (B, N, 1)

        # Radial pressure gradient projected onto wind direction
        dp_dr = self.ops.radial_gradient(p, u, v)  # (B, N, 1)

        # Physical radius from domain centre, scaled to metres
        domain_m  = self.cfg.domain_radius_deg * 111_000.0
        coords    = self.ops.graph.x_coords                    # (N, 2) in [-1,1]
        r_norm    = coords.norm(dim=-1)                        # (N,)  in [0, √2]
        r_phys    = (r_norm * domain_m).clamp(min=1e4)        # (N,) in metres
        r         = r_phys.view(1, -1, 1).to(s_tp1.device)   # (1, N, 1)

        R_wp = V.pow(2) / r + self.f * V + (1.0 / self.rho) * dp_dr
        return R_wp.pow(2).mean()

    # ── 5. Temporal continuity ─────────────────────────────────────────────────

    def temporal_continuity(self, s_t: Tensor, s_tp1: Tensor) -> Tensor:
        """
        Soft constraint: the state should not change faster than physically
        plausible.

        R_cont = ReLU( ‖s_{t+1} − s_t‖² / N − threshold )

        where threshold = (C · Δt)² / C_channels, C = 50 m/s.
        Returns 0 if the change is within physical bounds.
        """
        N          = s_t.shape[1]
        C_channels = s_t.shape[2]
        diff_sq    = (s_tp1 - s_t).pow(2).mean()
        C_scale    = 50.0                        # m/s representative scale
        threshold  = (C_scale * self.dt) ** 2 / C_channels
        return torch.relu(diff_sq - threshold)

    # ── 6. Kinetic-energy conservation ────────────────────────────────────────

    def energy(self, s_t: Tensor, s_tp1: Tensor) -> Tensor:
        """
        Inviscid kinetic energy balance:
          ∂E/∂t + ∂(E·u)/∂x + ∂(E·v)/∂y = 0
          E = ½(u² + v²)

        Returns: MSE(R_E).
        """
        u   = s_t[..., 0:1];   v   = s_t[..., 1:2]
        u1  = s_tp1[..., 0:1]; v1  = s_tp1[..., 1:2]

        E   = 0.5 * (u.pow(2) + v.pow(2))
        E1  = 0.5 * (u1.pow(2) + v1.pow(2))

        dEu_dx, _      = self.ops.gradient(E * u)
        _,      dEv_dy = self.ops.gradient(E * v)

        R_E = (E1 - E) / self.dt + dEu_dx + dEv_dy
        return R_E.pow(2).mean()

    # ── Aggregate ──────────────────────────────────────────────────────────────

    def all_residuals(
        self, s_t: Tensor, s_tp1: Tensor
    ) -> Dict[str, Tensor]:
        """
        Compute all six physics residuals and return as a named dict.

        Args:
            s_t   : (B, N, 7) current state
            s_tp1 : (B, N, 7) next state (ground truth or predicted)

        Returns:
            dict with keys: adv, diff, mass, wp, cont, nrg
        """
        return {
            "adv":  self.advection(s_t, s_tp1),
            "diff": self.diffusion(s_t, s_tp1),
            "mass": self.mass_conservation(s_t, s_tp1),   # predicted state
            "wp":   self.wind_pressure(s_t, s_tp1),       # predicted state
            "cont": self.temporal_continuity(s_t, s_tp1),
            "nrg":  self.energy(s_t, s_tp1),
        }
