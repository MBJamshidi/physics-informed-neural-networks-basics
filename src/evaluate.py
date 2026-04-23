# =============================================================================
# src/evaluate.py
#
# Evaluate the trained PINN against the numerical reference solution.
#
# Metrics:
#   MSE  — mean squared error
#   MAE  — mean absolute error
#   RelL2 — relative L2 error = ||u_pred - u_ref|| / ||u_ref||
# =============================================================================

import numpy as np
import torch
from src.model import PINN


def predict_full_grid(model: PINN,
                      eval_tensors: dict,
                      device: torch.device) -> np.ndarray:
    """
    Run the PINN over the entire (x, t) evaluation grid.

    Parameters
    ----------
    model        : trained PINN
    eval_tensors : dict from dataset.build_eval_tensors
    device       : torch device

    Returns
    -------
    U_pred : 2-D numpy array of shape (nt, nx)
    """
    model.eval()
    with torch.no_grad():
        X_eval = eval_tensors["X_eval"]
        T_eval = eval_tensors["T_eval"]
        u_flat = model(X_eval, T_eval)        # shape (nt*nx, 1)

    nt = eval_tensors["nt"]
    nx = eval_tensors["nx"]
    U_pred = u_flat.cpu().numpy().reshape(nt, nx)
    return U_pred


def compute_metrics(U_pred: np.ndarray, U_ref: np.ndarray) -> dict:
    """
    Compute error metrics between prediction and reference.

    Parameters
    ----------
    U_pred : predicted solution, shape (nt, nx)
    U_ref  : reference (numerical) solution, same shape

    Returns
    -------
    dict with keys: mse, mae, rel_l2
    """
    diff = U_pred - U_ref

    mse   = float(np.mean(diff ** 2))
    mae   = float(np.mean(np.abs(diff)))
    rel_l2 = float(np.linalg.norm(diff) / (np.linalg.norm(U_ref) + 1e-10))

    return {"mse": mse, "mae": mae, "rel_l2": rel_l2}


def print_metrics(metrics: dict) -> None:
    """Print evaluation metrics to console."""
    print("\n" + "=" * 40)
    print("  Evaluation Metrics")
    print("=" * 40)
    print(f"  MSE      : {metrics['mse']:.6e}")
    print(f"  MAE      : {metrics['mae']:.6e}")
    print(f"  Rel. L2  : {metrics['rel_l2']:.6e}")
    print("=" * 40 + "\n")
