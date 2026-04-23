# =============================================================================
# src/train.py
#
# Training loop for the wave-equation PINN.
#
# Steps each epoch:
#   1. Zero gradients
#   2. Compute total loss (data + physics + BC + IC)
#   3. Backpropagate
#   4. Adam optimizer step
#   5. Log and save periodically
# =============================================================================

import os
import csv
import time
import torch
from src.model  import PINN
from src.losses import total_loss


def train(model:      PINN,
          tensors:    dict,
          c:          float,
          epochs:     int,
          lr:         float,
          lam_data:   float,
          lam_phys:   float,
          lam_bc:     float,
          lam_ic:     float,
          log_every:  int,
          model_path: str,
          log_path:   str,
          device:     torch.device) -> list:
    """
    Train the PINN and return the full loss history.

    Parameters
    ----------
    model      : PINN instance
    tensors    : dict of tensors from dataset.build_training_tensors
    c          : wave speed
    epochs     : number of training epochs
    lr         : Adam learning rate
    lam_*      : loss weights
    log_every  : print frequency
    model_path : where to save the final model checkpoint
    log_path   : CSV file for loss history
    device     : torch device

    Returns
    -------
    history : list of dicts, one per epoch, with keys:
              epoch, total, data, phys, bc, ic
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Optionally use a learning-rate scheduler to reduce lr when training stalls
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=500, verbose=False
    )

    # Prepare CSV log file
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["epoch", "total", "data", "phys", "bc", "ic"])

    history = []
    t_start = time.time()

    print("=" * 70)
    print(f"{'Epoch':>7}  {'Total':>10}  {'Data':>10}  "
          f"{'Physics':>10}  {'BC':>10}  {'IC':>10}")
    print("=" * 70)

    for epoch in range(1, epochs + 1):

        model.train()
        optimizer.zero_grad()

        # --- compute all loss terms ---
        loss, l_data, l_phys, l_bc, l_ic = total_loss(
            model, tensors, c,
            lam_data, lam_phys, lam_bc, lam_ic
        )

        # --- backpropagation ---
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        # --- record history ---
        row = {
            "epoch": epoch,
            "total": loss.item(),
            "data":  l_data.item(),
            "phys":  l_phys.item(),
            "bc":    l_bc.item(),
            "ic":    l_ic.item()
        }
        history.append(row)
        writer.writerow([epoch, row["total"], row["data"],
                         row["phys"], row["bc"], row["ic"]])

        # --- console log ---
        if epoch % log_every == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"{epoch:>7}  "
                  f"{row['total']:>10.4e}  "
                  f"{row['data']:>10.4e}  "
                  f"{row['phys']:>10.4e}  "
                  f"{row['bc']:>10.4e}  "
                  f"{row['ic']:>10.4e}  "
                  f"  [{elapsed:.1f}s]")

    log_file.close()
    print("=" * 70)
    print(f"Training finished. Total time: {time.time() - t_start:.1f}s")

    # --- save model checkpoint ---
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"[Train] Model saved  →  {model_path}")

    return history
