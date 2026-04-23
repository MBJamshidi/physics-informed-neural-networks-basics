# =============================================================================
# src/losses.py
#
# Four loss components for the wave-equation PINN.
#
# Total loss:
#   L = λ_data * L_data  +  λ_phys * L_phys  +  λ_bc * L_bc  +  λ_ic * L_ic
#
# Component definitions
# ─────────────────────
# L_data  : MSE between PINN prediction and numerical solution at sampled pts
# L_phys  : MSE of the PDE residual (u_tt - c²*u_xx) at collocation pts
# L_bc    : MSE of PINN output at x=0 and x=1 (should be 0 — Dirichlet walls)
# L_ic    : MSE of displacement u(x,0) vs sin(πx), plus MSE of u_t(x,0) vs 0
# =============================================================================

import torch
import torch.nn as nn
from src.model import PINN, compute_pde_residual, compute_ic_velocity

mse = nn.MSELoss()


def data_loss(model: PINN,
              X_data: torch.Tensor,
              T_data: torch.Tensor,
              U_data: torch.Tensor) -> torch.Tensor:
    """
    Supervised data loss — penalises deviation from numerical solution.

    L_data = mean( (u_pred(x_i, t_i) - u_num(x_i, t_i))^2 )
    """
    u_pred = model(X_data, T_data)
    return mse(u_pred, U_data)


def physics_loss(model: PINN,
                 X_col: torch.Tensor,
                 T_col: torch.Tensor,
                 c:     float) -> torch.Tensor:
    """
    PDE residual loss — enforces the wave equation at collocation points.

    L_phys = mean( (u_tt - c^2 * u_xx)^2 )
    """
    residual = compute_pde_residual(model, X_col, T_col, c)
    return mse(residual, torch.zeros_like(residual))


def bc_loss(model: PINN,
            X_bc_left:  torch.Tensor,
            X_bc_right: torch.Tensor,
            T_bc:       torch.Tensor,
            U_bc:       torch.Tensor) -> torch.Tensor:
    """
    Boundary condition loss — enforces u = 0 at both walls.

    L_bc = mean( u(0, t)^2 )  +  mean( u(1, t)^2 )
    """
    u_left  = model(X_bc_left,  T_bc)
    u_right = model(X_bc_right, T_bc)
    return mse(u_left, U_bc) + mse(u_right, U_bc)


def ic_loss(model: PINN,
            X_ic:  torch.Tensor,
            T_ic:  torch.Tensor,
            U_ic:  torch.Tensor,
            Ut_ic: torch.Tensor) -> torch.Tensor:
    """
    Initial condition loss — enforces both displacement and velocity at t=0.

    L_ic = mean( (u(x,0) - sin(πx))^2 )  +  mean( (u_t(x,0) - 0)^2 )
    """
    # Displacement IC: u(x, 0) = sin(πx)
    u_pred  = model(X_ic, T_ic)
    loss_u  = mse(u_pred, U_ic)

    # Velocity IC: u_t(x, 0) = 0
    ut_pred = compute_ic_velocity(model, X_ic, T_ic)
    loss_ut = mse(ut_pred, Ut_ic)

    return loss_u + loss_ut


def total_loss(model: PINN,
               tensors: dict,
               c:          float,
               lam_data:   float,
               lam_phys:   float,
               lam_bc:     float,
               lam_ic:     float) -> tuple:
    """
    Compute all four loss components and the weighted total.

    Returns
    -------
    total, l_data, l_phys, l_bc, l_ic  — all scalar tensors
    """
    l_data = data_loss(
        model,
        tensors["X_data"], tensors["T_data"], tensors["U_data"]
    )
    l_phys = physics_loss(
        model,
        tensors["X_col"], tensors["T_col"], c
    )
    l_bc = bc_loss(
        model,
        tensors["X_bc_left"], tensors["X_bc_right"],
        tensors["T_bc"], tensors["U_bc"]
    )
    l_ic = ic_loss(
        model,
        tensors["X_ic"], tensors["T_ic"],
        tensors["U_ic"], tensors["Ut_ic"]
    )

    total = (lam_data * l_data
             + lam_phys * l_phys
             + lam_bc   * l_bc
             + lam_ic   * l_ic)

    return total, l_data, l_phys, l_bc, l_ic
