# =============================================================================
# src/model.py
#
# Physics-Informed Neural Network (PINN) for the 1D wave equation.
#
# Architecture:
#   Input  : (x, t)  — 2 features
#   Hidden : N_LAYERS fully-connected layers, each with HIDDEN_DIM neurons
#            and Tanh activation (smooth enough for second-order derivatives)
#   Output : u(x, t) — scalar prediction
#
# Why Tanh?
#   The PDE involves second-order derivatives (u_tt, u_xx).  Tanh is infinitely
#   differentiable, which makes autograd stable through multiple derivative
#   passes. ReLU's second derivative is zero almost everywhere — unsuitable.
#
# Derivative computation:
#   We use torch.autograd.grad with create_graph=True so the computational
#   graph is preserved for higher-order derivatives (needed for u_tt, u_xx).
# =============================================================================

import torch
import torch.nn as nn


class PINN(nn.Module):
    """
    Simple fully-connected PINN that maps (x, t) → u(x, t).

    Parameters
    ----------
    input_dim  : number of input features (2 for wave equation: x, t)
    hidden_dim : width of each hidden layer
    n_layers   : number of hidden layers
    output_dim : number of outputs (1: scalar u)
    """

    def __init__(self,
                 input_dim:  int = 2,
                 hidden_dim: int = 64,
                 n_layers:   int = 5,
                 output_dim: int = 1):
        super().__init__()

        # Build a sequential stack of Linear + Tanh layers
        layers = []

        # First layer: input → hidden
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        # Intermediate hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # Output layer: hidden → scalar  (no activation — unbounded output)
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

        # Xavier / Glorot initialisation keeps gradients well-scaled at start
        self._init_weights()

    def _init_weights(self):
        """Apply Xavier uniform initialisation to all Linear layers."""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : tensor of shape (N, 1) — spatial coordinates
        t : tensor of shape (N, 1) — temporal coordinates

        Returns
        -------
        u : tensor of shape (N, 1) — predicted solution
        """
        # Concatenate spatial and temporal inputs along the feature dimension
        xt = torch.cat([x, t], dim=1)   # shape (N, 2)
        return self.net(xt)              # shape (N, 1)


# ---------------------------------------------------------------------------
# Derivative utilities using PyTorch autograd
# ---------------------------------------------------------------------------

def grad(output: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """
    Compute the gradient of `output` with respect to `inp` using autograd.

    create_graph=True  — keeps the computation graph so we can differentiate
                         again (needed for second-order terms like u_xx).
    retain_graph=True  — keep the graph alive for multiple backward passes.
    """
    return torch.autograd.grad(
        output, inp,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True
    )[0]


def compute_pde_residual(model: PINN,
                         x_col: torch.Tensor,
                         t_col: torch.Tensor,
                         c: float) -> torch.Tensor:
    """
    Evaluate the wave equation PDE residual at collocation points.

    PDE:   u_tt  =  c^2 * u_xx
    Residual:  r = u_tt - c^2 * u_xx   (should be 0 if PDE is satisfied)

    Parameters
    ----------
    model  : PINN instance
    x_col  : collocation x-coordinates, shape (N, 1), requires_grad=True
    t_col  : collocation t-coordinates, shape (N, 1), requires_grad=True
    c      : wave speed (scalar)

    Returns
    -------
    residual : tensor of shape (N, 1)
    """
    # Forward pass — predict u at collocation points
    u = model(x_col, t_col)

    # First-order derivatives
    u_t  = grad(u, t_col)     # ∂u/∂t
    u_x  = grad(u, x_col)     # ∂u/∂x  (needed to get u_xx)

    # Second-order derivatives
    u_tt = grad(u_t, t_col)   # ∂²u/∂t²
    u_xx = grad(u_x, x_col)   # ∂²u/∂x²

    # PDE residual: r = u_tt - c^2 * u_xx   (must be ≈ 0)
    residual = u_tt - (c ** 2) * u_xx

    return residual


def compute_ic_velocity(model: PINN,
                        x_ic: torch.Tensor,
                        t_ic: torch.Tensor) -> torch.Tensor:
    """
    Compute u_t(x, 0) at initial condition points.
    Needed to enforce the zero-velocity IC: u_t(x, 0) = 0.
    """
    u = model(x_ic, t_ic)
    u_t = grad(u, t_ic)
    return u_t
