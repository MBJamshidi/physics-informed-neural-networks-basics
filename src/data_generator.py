# =============================================================================
# src/data_generator.py
#
# Generates synthetic data for the 1D wave equation using an explicit
# finite-difference (FD) scheme.
#
# PDE:   u_tt = c^2 * u_xx          x in [0,1],  t in [0, T]
# IC:    u(x, 0)   = sin(pi*x)      initial displacement
#        u_t(x, 0) = 0              initial velocity (at rest)
# BC:    u(0, t)   = 0              left wall (Dirichlet)
#        u(1, t)   = 0              right wall (Dirichlet)
#
# Exact analytical solution for these conditions:
#        u(x, t) = sin(pi*x) * cos(c*pi*t)
#
# The FD solver uses the standard second-order explicit leapfrog scheme:
#   u[n+1] = 2*u[n] - u[n-1] + r^2 * (u[n,i+1] - 2*u[n,i] + u[n,i-1])
# where r = c * dt / dx  (CFL number — must be <= 1 for stability)
# =============================================================================

import os
import numpy as np


def solve_wave_fd(nx: int, nt: int, c: float,
                  x_min: float, x_max: float,
                  t_min: float, t_max: float,
                  seed: int = 42) -> dict:
    """
    Solve the 1D wave equation using an explicit finite-difference scheme.

    Parameters
    ----------
    nx    : number of spatial grid points
    nt    : number of temporal grid points
    c     : wave propagation speed
    x_min, x_max : spatial domain
    t_min, t_max : temporal domain
    seed  : random seed (used when this data is later sampled for training)

    Returns
    -------
    dict with keys:
        x   : 1-D array of spatial grid  (shape: nx)
        t   : 1-D array of temporal grid (shape: nt)
        U   : 2-D solution array         (shape: nt x nx)
              U[i, j] = u(x_j, t_i)
        U_exact : 2-D analytical solution (same shape, for verification)
    """
    np.random.seed(seed)

    # --- build spatial and temporal grids ---
    x = np.linspace(x_min, x_max, nx)    # shape (nx,)
    t = np.linspace(t_min, t_max, nt)    # shape (nt,)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # --- CFL stability check ---
    # The explicit leapfrog scheme is stable only if r = c*dt/dx <= 1.
    r = c * dt / dx
    if r > 1.0:
        raise ValueError(
            f"CFL number r={r:.4f} > 1 — scheme is unstable. "
            f"Increase nt or decrease nx."
        )
    print(f"[FD Solver] Grid: nx={nx}, nt={nt}  |  dx={dx:.5f}, dt={dt:.6f}")
    print(f"[FD Solver] CFL number r = c*dt/dx = {r:.4f}  (stable: r <= 1)")

    r2 = r ** 2  # r^2 used in the update formula

    # --- allocate solution array ---
    U = np.zeros((nt, nx))   # U[time_index, space_index]

    # --- Initial condition: u(x, 0) = sin(pi*x) ---
    U[0, :] = np.sin(np.pi * x)

    # --- Boundary conditions: u(0,t) = u(1,t) = 0  (held throughout loop) ---
    # (interior indices: 1 to nx-2)

    # --- First time step using zero initial velocity u_t(x,0) = 0 ---
    # Derived from Taylor expansion:
    #   u(x, dt) ≈ u(x,0) + dt*u_t(x,0) + 0.5*dt^2*u_tt(x,0)
    # Since u_t = 0 and u_tt = c^2 * u_xx at t=0:
    U[1, 1:-1] = (U[0, 1:-1]
                  + 0.5 * r2 * (U[0, 2:] - 2*U[0, 1:-1] + U[0, :-2]))
    # enforce BC at the first step
    U[1, 0] = 0.0
    U[1, -1] = 0.0

    # --- Main time-stepping loop (leapfrog scheme) ---
    # u[n+1,i] = 2*u[n,i] - u[n-1,i] + r^2*(u[n,i+1] - 2*u[n,i] + u[n,i-1])
    for n in range(1, nt - 1):
        U[n+1, 1:-1] = (2.0 * U[n, 1:-1]
                        - U[n-1, 1:-1]
                        + r2 * (U[n, 2:] - 2*U[n, 1:-1] + U[n, :-2]))
        # Dirichlet boundary conditions (walls stay at 0)
        U[n+1, 0]  = 0.0
        U[n+1, -1] = 0.0

    # --- Analytical solution for verification ---
    # u_exact(x, t) = sin(pi*x) * cos(c*pi*t)
    T_grid, X_grid = np.meshgrid(t, x, indexing='ij')   # both (nt, nx)
    U_exact = np.sin(np.pi * X_grid) * np.cos(c * np.pi * T_grid)

    # Quick sanity check — compare FD vs exact
    max_err = np.max(np.abs(U - U_exact))
    print(f"[FD Solver] Max |FD - exact| over full grid = {max_err:.6f}")

    return {"x": x, "t": t, "U": U, "U_exact": U_exact}


def save_dataset(solution: dict, save_path: str) -> None:
    """
    Save the numerical solution to a compressed .npz file.

    Parameters
    ----------
    solution  : dict returned by solve_wave_fd
    save_path : full path including filename, e.g. data/raw/wave_solution.npz
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(
        save_path,
        x=solution["x"],
        t=solution["t"],
        U=solution["U"],
        U_exact=solution["U_exact"]
    )
    print(f"[Data] Dataset saved  →  {save_path}")


def load_dataset(load_path: str) -> dict:
    """
    Load the numerical solution from a .npz file.

    Returns a dict with keys: x, t, U, U_exact
    """
    data = np.load(load_path)
    print(f"[Data] Dataset loaded ←  {load_path}")
    return {
        "x":       data["x"],
        "t":       data["t"],
        "U":       data["U"],
        "U_exact": data["U_exact"]
    }
