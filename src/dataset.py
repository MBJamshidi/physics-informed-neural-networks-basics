# =============================================================================
# src/dataset.py
#
# Converts the raw numerical solution grid into the tensors that the PINN
# training loop needs:
#
#   • data points     — (x, t, u_numerical) sampled from interior of the grid
#   • collocation pts — (x, t) randomly distributed in the domain for PDE loss
#   • IC points       — (x, 0) for initial displacement & velocity enforcement
#   • BC points       — (0, t) and (1, t) for Dirichlet wall conditions
#
# All tensors are returned with requires_grad=True where autograd is needed.
# =============================================================================

import numpy as np
import torch


def build_training_tensors(solution: dict,
                           n_data:   int,
                           n_colloc: int,
                           n_bc:     int,
                           n_ic:     int,
                           device:   torch.device,
                           seed:     int = 42) -> dict:
    """
    Sample training tensors from the numerical solution grid.

    Parameters
    ----------
    solution  : dict from data_generator (keys: x, t, U, U_exact)
    n_data    : number of supervised data points (data loss)
    n_colloc  : number of interior collocation points (physics loss)
    n_bc      : number of boundary points per wall (BC loss)
    n_ic      : number of initial condition points  (IC loss)
    device    : torch device (cpu or cuda)
    seed      : random seed for reproducibility

    Returns
    -------
    dict of torch tensors ready for training
    """
    rng = np.random.default_rng(seed)

    x_grid = solution["x"]   # shape (nx,)
    t_grid = solution["t"]   # shape (nt,)
    U_grid = solution["U"]   # shape (nt, nx)  — numerical solution
    nx = len(x_grid)
    nt = len(t_grid)

    # ------------------------------------------------------------------ #
    # 1. DATA POINTS — random (i, j) pairs from the full solution grid    #
    #    We skip the very first time step (t=0) and boundaries in x here  #
    #    because IC and BC are handled separately.                         #
    # ------------------------------------------------------------------ #
    ti_data = rng.integers(1, nt, size=n_data)       # time indices
    xi_data = rng.integers(1, nx - 1, size=n_data)   # space indices (interior)

    x_data = x_grid[xi_data].astype(np.float32)
    t_data = t_grid[ti_data].astype(np.float32)
    u_data = U_grid[ti_data, xi_data].astype(np.float32)

    X_data = _to_tensor(x_data, device, grad=False)
    T_data = _to_tensor(t_data, device, grad=False)
    U_data = _to_tensor(u_data, device, grad=False)

    # ------------------------------------------------------------------ #
    # 2. COLLOCATION POINTS — uniform random in (x,t) domain             #
    #    These are NOT tied to the numerical grid — they can be anywhere. #
    #    The PINN must satisfy the PDE at these points.                   #
    # ------------------------------------------------------------------ #
    x_col = rng.uniform(0.0, 1.0, size=n_colloc).astype(np.float32)
    t_col = rng.uniform(0.0, 1.0, size=n_colloc).astype(np.float32)

    # requires_grad=True so PyTorch can differentiate u w.r.t. x and t
    X_col = _to_tensor(x_col, device, grad=True)
    T_col = _to_tensor(t_col, device, grad=True)

    # ------------------------------------------------------------------ #
    # 3. INITIAL CONDITION POINTS — t = 0, x uniform in [0, 1]           #
    #    Two conditions must hold:                                         #
    #      u(x, 0)   = sin(pi*x)   (displacement)                         #
    #      u_t(x, 0) = 0           (zero initial velocity)                 #
    # ------------------------------------------------------------------ #
    x_ic = rng.uniform(0.0, 1.0, size=n_ic).astype(np.float32)
    t_ic = np.zeros(n_ic, dtype=np.float32)

    u_ic_true = np.sin(np.pi * x_ic)   # analytical IC displacement
    ut_ic_true = np.zeros(n_ic, dtype=np.float32)  # IC velocity = 0

    X_ic  = _to_tensor(x_ic,      device, grad=True)
    T_ic  = _to_tensor(t_ic,      device, grad=True)
    U_ic  = _to_tensor(u_ic_true, device, grad=False)
    Ut_ic = _to_tensor(ut_ic_true, device, grad=False)

    # ------------------------------------------------------------------ #
    # 4. BOUNDARY CONDITION POINTS — x=0 and x=1, t uniform in [0, T]   #
    #    u(0, t) = 0  and  u(1, t) = 0  (Dirichlet walls)               #
    # ------------------------------------------------------------------ #
    t_bc = rng.uniform(0.0, 1.0, size=n_bc).astype(np.float32)

    x_left  = np.zeros(n_bc, dtype=np.float32)   # x = 0
    x_right = np.ones(n_bc,  dtype=np.float32)   # x = 1
    u_bc_val = np.zeros(n_bc, dtype=np.float32)  # u = 0 at both walls

    X_bc_left  = _to_tensor(x_left,   device, grad=True)
    X_bc_right = _to_tensor(x_right,  device, grad=True)
    T_bc       = _to_tensor(t_bc,     device, grad=True)
    U_bc       = _to_tensor(u_bc_val, device, grad=False)

    return {
        # supervised data
        "X_data": X_data, "T_data": T_data, "U_data": U_data,
        # physics collocation
        "X_col": X_col, "T_col": T_col,
        # initial conditions
        "X_ic": X_ic, "T_ic": T_ic, "U_ic": U_ic, "Ut_ic": Ut_ic,
        # boundary conditions
        "X_bc_left": X_bc_left, "X_bc_right": X_bc_right,
        "T_bc": T_bc, "U_bc": U_bc,
    }


def build_eval_tensors(solution: dict, device: torch.device) -> dict:
    """
    Build a dense evaluation grid covering the full (x, t) domain.
    Used after training to compare PINN output against numerical solution.

    Returns
    -------
    dict with X_eval, T_eval (flat tensors) and grid shapes for plotting
    """
    x_grid = solution["x"].astype(np.float32)
    t_grid = solution["t"].astype(np.float32)
    nx = len(x_grid)
    nt = len(t_grid)

    # meshgrid: rows = time, cols = space  →  flatten to (nt*nx,) vectors
    T_mg, X_mg = np.meshgrid(t_grid, x_grid, indexing='ij')  # (nt, nx)
    X_flat = X_mg.ravel().astype(np.float32)
    T_flat = T_mg.ravel().astype(np.float32)

    X_eval = _to_tensor(X_flat, device, grad=False)
    T_eval = _to_tensor(T_flat, device, grad=False)

    return {"X_eval": X_eval, "T_eval": T_eval, "nx": nx, "nt": nt}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _to_tensor(arr: np.ndarray, device: torch.device,
               grad: bool = False) -> torch.Tensor:
    """Convert 1-D numpy array to a column tensor (n, 1) on device."""
    t = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(1)
    t.requires_grad_(grad)
    return t
