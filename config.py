# =============================================================================
# config.py
# Central configuration for the 1D Wave Equation PINN project.
# All hyperparameters, domain settings, and file paths live here.
# Change values here rather than hunting through the source files.
# =============================================================================

import os

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42  # fixed random seed for numpy and torch

# ---------------------------------------------------------------------------
# Physical parameters — 1D wave equation: u_tt = c^2 * u_xx
# ---------------------------------------------------------------------------
WAVE_SPEED = 1.0   # c  (propagation speed)
X_MIN = 0.0        # left boundary
X_MAX = 1.0        # right boundary
T_MIN = 0.0        # start time
T_MAX = 1.0        # end time (keep ≤ 1 for light CPU training)

# ---------------------------------------------------------------------------
# Numerical solver (finite difference) settings
# ---------------------------------------------------------------------------
NX = 200           # spatial grid points (x)
NT = 500           # temporal grid points (t)

# ---------------------------------------------------------------------------
# PINN architecture
# ---------------------------------------------------------------------------
INPUT_DIM  = 2     # network input: (x, t)
HIDDEN_DIM = 64    # neurons per hidden layer
N_LAYERS   = 5     # number of hidden layers
OUTPUT_DIM = 1     # network output: u(x, t)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
EPOCHS            = 8000   # total training epochs
LEARNING_RATE     = 1e-3   # Adam optimizer learning rate
N_DATA_POINTS     = 2000   # sampled points from numerical solution (data loss)
N_COLLOC_POINTS   = 5000   # collocation points for physics loss (interior)
N_BC_POINTS       = 500    # boundary condition points (each boundary)
N_IC_POINTS       = 500    # initial condition points (u and u_t)

# Loss weights — tune these to balance the four loss terms
LAMBDA_DATA = 1.0   # weight for supervised data loss
LAMBDA_PHYS = 1.0   # weight for PDE residual loss
LAMBDA_BC   = 10.0  # weight for boundary condition loss (enforce strongly)
LAMBDA_IC   = 10.0  # weight for initial condition loss (enforce strongly)

LOG_EVERY   = 500   # print training progress every N epochs

# ---------------------------------------------------------------------------
# Paths  (all relative to project root so Windows & Linux both work)
# ---------------------------------------------------------------------------
ROOT_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR    = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROC_DIR   = os.path.join(ROOT_DIR, "data", "processed")
OUTPUT_FIG_DIR  = os.path.join(ROOT_DIR, "outputs", "figures")
OUTPUT_MDL_DIR  = os.path.join(ROOT_DIR, "outputs", "models")
OUTPUT_LOG_DIR  = os.path.join(ROOT_DIR, "outputs", "logs")

DATASET_FILE    = os.path.join(DATA_RAW_DIR, "wave_solution.npz")
MODEL_FILE      = os.path.join(OUTPUT_MDL_DIR, "pinn_wave.pth")
LOSS_LOG_FILE   = os.path.join(OUTPUT_LOG_DIR, "loss_history.csv")
