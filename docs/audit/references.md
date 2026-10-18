# Reference cases

Ghia, Ghia, and Shin, Journal of Computational Physics 48 (1982), lid-driven cavity, Re = 100.
The oracle uses a 32×32×1 upwind MAC grid, integrated to t = 12. Interior centerline samples are required to stay within 0.05 of the published u values. The sample at y = 0.9531 is allowed 0.25 because that grid does not resolve the lid layer. This is not a grid-converged Ghia study.

Heat: a uniform temperature with Boussinesq buoyancy must stay at rest. A Dirichlet wall-to-wall drop must relax toward a straight line. Those two cases are the only checked heat claims.

Forces on a voxel block are finite and the step stays divergence-free. No drag coefficient is claimed.
