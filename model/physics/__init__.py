"""
model.physics — Physics-Informed Graph Neural Operator (PI-GNO)

Public API
----------
  PIGNOConfig         : All hyperparameters (grid, GNO, FNO, physics, training)
  PIGNOModel          : Full GNO+FNO model with field and track heads
  PhysicsResiduals    : Six PDE residual computations (advection, diffusion,
                        mass, wind-pressure, continuity, energy)
  PhysicsInformedLoss : Weighted physics + data loss with warm-up schedule
  PIGNOTrainer        : Training loop with stability analysis and checkpointing

Quick start
-----------
  # Demo (CPU, ~4 min)
  python -m model.physics.train --demo

  # Custom
  python -m model.physics.train --epochs 50 --grid-size 33

Architecture
------------
  Input (B, N, 7)
    └─ Lifting → d_v
    └─ GNO × n_gno_layers   [local, physical-space kernel integral]
    └─ Reshape → (B, d_v, H, W)
    └─ FNO × n_fno_layers   [global, Fourier-space spectral operator]
    └─ Reshape → (B, N, d_v)
    ├─ FieldHead → (B, N, 7)   predicted field increments Δs
    └─ TrackHead → (B, 2)      predicted track displacement [Δlat, Δlon]
"""

from .config      import PIGNOConfig
from .architecture import PIGNOModel
from .physics_kernels import PhysicsResiduals
from .losses      import PhysicsInformedLoss
from .train       import PIGNOTrainer

__all__ = [
    "PIGNOConfig",
    "PIGNOModel",
    "PhysicsResiduals",
    "PhysicsInformedLoss",
    "PIGNOTrainer",
]
