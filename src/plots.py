# =============================================================================
# src/plots.py
#
# All plotting utilities for the wave-equation PINN project.
#
# Saves figures to outputs/figures/ (path passed as argument).
# Uses non-interactive Agg backend — safe for servers / headless runs.
# =============================================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive: saves files, doesn't open GUI
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Numerical solution visualisation
# ---------------------------------------------------------------------------

def plot_numerical_solution(x: np.ndarray,
                            t: np.ndarray,
                            U: np.ndarray,
                            save_dir: str) -> None:
    """
    Plot the numerically generated wave solution as a heatmap and
    snapshots at several time instants.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # --- Heatmap ---
    im = axes[0].imshow(
        U, aspect="auto", origin="lower",
        extent=[x[0], x[-1], t[0], t[-1]],
        cmap="RdBu_r"
    )
    plt.colorbar(im, ax=axes[0], label="u(x, t)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("t")
    axes[0].set_title("Numerical Solution — u(x, t)")

    # --- Snapshots at 5 time instants ---
    nt = len(t)
    snap_indices = [0, nt//8, nt//4, nt//2, nt - 1]
    colors = plt.cm.plasma(np.linspace(0, 1, len(snap_indices)))
    for idx, color in zip(snap_indices, colors):
        axes[1].plot(x, U[idx, :], color=color, label=f"t={t[idx]:.2f}")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("u(x, t)")
    axes[1].set_title("Snapshots at Selected Times")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "numerical_solution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Numerical solution saved  →  {path}")


# ---------------------------------------------------------------------------
# 2. Training loss history
# ---------------------------------------------------------------------------

def plot_loss_history(history: list, save_dir: str) -> None:
    """
    Plot total loss and each component over training epochs.
    Uses a log-scale y-axis to reveal convergence across orders of magnitude.
    """
    os.makedirs(save_dir, exist_ok=True)

    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Total loss
    axes[0].semilogy(epochs, [r["total"] for r in history], color="black", lw=1.5)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (log scale)")
    axes[0].set_title("Total Loss"); axes[0].grid(True, alpha=0.3)

    # Individual components
    components = {"Data": "data", "Physics": "phys", "BC": "bc", "IC": "ic"}
    palette = ["steelblue", "darkorange", "green", "crimson"]
    for (label, key), color in zip(components.items(), palette):
        axes[1].semilogy(epochs, [r[key] for r in history],
                         label=label, color=color, lw=1.2)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss (log scale)")
    axes[1].set_title("Loss Components"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "loss_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Loss history saved  →  {path}")


# ---------------------------------------------------------------------------
# 3. Comparison: numerical vs PINN vs absolute error
# ---------------------------------------------------------------------------

def plot_comparison(x: np.ndarray,
                    t: np.ndarray,
                    U_ref:  np.ndarray,
                    U_pred: np.ndarray,
                    save_dir: str) -> None:
    """
    Side-by-side heatmaps:  reference | PINN prediction | absolute error.
    """
    os.makedirs(save_dir, exist_ok=True)
    abs_err = np.abs(U_pred - U_ref)

    extent = [x[0], x[-1], t[0], t[-1]]
    vmin = min(U_ref.min(), U_pred.min())
    vmax = max(U_ref.max(), U_pred.max())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    titles    = ["Numerical (Reference)", "PINN Prediction", "Absolute Error"]
    arrays    = [U_ref, U_pred, abs_err]
    cmaps     = ["RdBu_r", "RdBu_r", "hot_r"]

    for ax, title, arr, cmap in zip(axes, titles, arrays, cmaps):
        kwargs = dict(aspect="auto", origin="lower",
                      extent=extent, cmap=cmap)
        if title != "Absolute Error":
            kwargs.update(vmin=vmin, vmax=vmax)
        im = ax.imshow(arr, **kwargs)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("x"); ax.set_ylabel("t")
        ax.set_title(title)

    plt.suptitle("PINN vs Numerical Solution — 1D Wave Equation",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_heatmaps.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Comparison heatmaps saved  →  {path}")


# ---------------------------------------------------------------------------
# 4. Snapshot comparison at a fixed time
# ---------------------------------------------------------------------------

def plot_snapshots(x: np.ndarray,
                   t: np.ndarray,
                   U_ref:  np.ndarray,
                   U_pred: np.ndarray,
                   save_dir: str,
                   n_snaps: int = 4) -> None:
    """
    Plot u(x) profiles at several time slices:
    reference (solid) vs PINN prediction (dashed).
    """
    os.makedirs(save_dir, exist_ok=True)
    nt = len(t)
    indices = np.linspace(0, nt - 1, n_snaps, dtype=int)

    fig, axes = plt.subplots(1, n_snaps, figsize=(4 * n_snaps, 4), sharey=True)
    for ax, idx in zip(axes, indices):
        ax.plot(x, U_ref[idx],  label="Numerical", color="steelblue", lw=2)
        ax.plot(x, U_pred[idx], label="PINN",      color="darkorange",
                lw=2, linestyle="--")
        ax.set_title(f"t = {t[idx]:.2f}")
        ax.set_xlabel("x")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("u(x, t)")

    plt.suptitle("Snapshots: PINN vs Numerical", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "snapshots.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Snapshots saved  →  {path}")
