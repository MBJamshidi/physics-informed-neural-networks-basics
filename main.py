# =============================================================================
# main.py
#
# End-to-end entry point for the 1D Wave Equation PINN project.
#
# Pipeline:
#   1. Parse arguments (override config values from command line if needed)
#   2. Set random seeds and create output directories
#   3. Generate / load numerical solution (finite difference)
#   4. Plot numerical solution
#   5. Build training tensors (data, collocation, IC, BC)
#   6. Instantiate PINN model
#   7. Train (data loss + physics loss + BC loss + IC loss)
#   8. Plot training loss curves
#   9. Evaluate on the full grid and print metrics
#  10. Save comparison plots
#
# Usage examples
# --------------
#   Full run (generate data + train + evaluate):
#       python main.py
#
#   Skip data generation (reuse existing dataset):
#       python main.py --skip-datagen
#
#   Quick smoke-test with fewer epochs:
#       python main.py --epochs 500
# =============================================================================

import argparse
import sys
import os

# Make sure the project root is on the Python path when running on Windows
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.utils         import set_seed, get_device, make_output_dirs, count_parameters
from src.data_generator import solve_wave_fd, save_dataset, load_dataset
from src.dataset        import build_training_tensors, build_eval_tensors
from src.model          import PINN
from src.train          import train
from src.evaluate       import predict_full_grid, compute_metrics, print_metrics
from src.plots          import (plot_numerical_solution, plot_loss_history,
                                plot_comparison, plot_snapshots)


# ---------------------------------------------------------------------------
# Argument parser — lets you override key config values without editing files
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Wave Equation PINN")
    p.add_argument("--skip-datagen", action="store_true",
                   help="Load existing dataset instead of regenerating it")
    p.add_argument("--epochs",  type=int,   default=config.EPOCHS)
    p.add_argument("--lr",      type=float, default=config.LEARNING_RATE)
    p.add_argument("--seed",    type=int,   default=config.SEED)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- 1. Reproducibility and output directories ---
    set_seed(args.seed)
    make_output_dirs(
        config.DATA_RAW_DIR,
        config.DATA_PROC_DIR,
        config.OUTPUT_FIG_DIR,
        config.OUTPUT_MDL_DIR,
        config.OUTPUT_LOG_DIR,
    )

    device = get_device()

    print("\n" + "=" * 60)
    print("  Physics-Informed Neural Network — 1D Wave Equation")
    print("=" * 60)
    print(f"  PDE  : u_tt = c^2 * u_xx")
    print(f"  c    : {config.WAVE_SPEED}")
    print(f"  x    : [{config.X_MIN}, {config.X_MAX}]")
    print(f"  t    : [{config.T_MIN}, {config.T_MAX}]")
    print(f"  IC   : u(x,0) = sin(πx),  u_t(x,0) = 0")
    print(f"  BC   : u(0,t) = u(1,t) = 0")
    print("=" * 60 + "\n")

    # --- 2. Generate or load numerical solution ---
    if args.skip_datagen and os.path.exists(config.DATASET_FILE):
        solution = load_dataset(config.DATASET_FILE)
    else:
        print("[Step 1] Generating numerical solution (finite difference)...")
        solution = solve_wave_fd(
            nx=config.NX, nt=config.NT, c=config.WAVE_SPEED,
            x_min=config.X_MIN, x_max=config.X_MAX,
            t_min=config.T_MIN, t_max=config.T_MAX,
            seed=args.seed
        )
        save_dataset(solution, config.DATASET_FILE)

    # --- 3. Plot numerical solution ---
    print("\n[Step 2] Plotting numerical solution...")
    plot_numerical_solution(
        solution["x"], solution["t"], solution["U"],
        save_dir=config.OUTPUT_FIG_DIR
    )

    # --- 4. Build training tensors ---
    print("\n[Step 3] Building training tensors...")
    tensors = build_training_tensors(
        solution,
        n_data   = config.N_DATA_POINTS,
        n_colloc = config.N_COLLOC_POINTS,
        n_bc     = config.N_BC_POINTS,
        n_ic     = config.N_IC_POINTS,
        device   = device,
        seed     = args.seed
    )

    # --- 5. Instantiate PINN ---
    print("\n[Step 4] Building PINN model...")
    model = PINN(
        input_dim  = config.INPUT_DIM,
        hidden_dim = config.HIDDEN_DIM,
        n_layers   = config.N_LAYERS,
        output_dim = config.OUTPUT_DIM
    ).to(device)
    print(f"  Trainable parameters: {count_parameters(model):,}")

    # --- 6. Train ---
    print(f"\n[Step 5] Training for {args.epochs} epochs...")
    history = train(
        model      = model,
        tensors    = tensors,
        c          = config.WAVE_SPEED,
        epochs     = args.epochs,
        lr         = args.lr,
        lam_data   = config.LAMBDA_DATA,
        lam_phys   = config.LAMBDA_PHYS,
        lam_bc     = config.LAMBDA_BC,
        lam_ic     = config.LAMBDA_IC,
        log_every  = config.LOG_EVERY,
        model_path = config.MODEL_FILE,
        log_path   = config.LOSS_LOG_FILE,
        device     = device
    )

    # --- 7. Plot loss curves ---
    print("\n[Step 6] Saving loss history plot...")
    plot_loss_history(history, save_dir=config.OUTPUT_FIG_DIR)

    # --- 8. Evaluate ---
    print("\n[Step 7] Evaluating on full grid...")
    eval_tensors = build_eval_tensors(solution, device)
    U_pred = predict_full_grid(model, eval_tensors, device)

    metrics = compute_metrics(U_pred, solution["U"])
    print_metrics(metrics)

    # --- 9. Comparison plots ---
    print("[Step 8] Saving comparison and snapshot plots...")
    plot_comparison(
        solution["x"], solution["t"],
        U_ref=solution["U"], U_pred=U_pred,
        save_dir=config.OUTPUT_FIG_DIR
    )
    plot_snapshots(
        solution["x"], solution["t"],
        U_ref=solution["U"], U_pred=U_pred,
        save_dir=config.OUTPUT_FIG_DIR
    )

    print("\n✓  All done. Outputs are in:", os.path.abspath("outputs"))


if __name__ == "__main__":
    main()
