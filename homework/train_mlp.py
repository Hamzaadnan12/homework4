import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.utils.tensorboard as tb

from models import MLPPlanner
from metrics import compute_errors  # Ensure this function is defined in metrics.py
from datasets.road_dataset import load_data  # Assuming you've created a dataset loading function


def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Set up logging directory
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Initialize MLPPlanner model
    model = MLPPlanner(**kwargs).to(device)
    model.train()

    # Load training and validation data (Make sure your dataset is correctly loaded)
    train_data = load_data("../drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("../drive_data/val", shuffle=False)

    # Loss function and optimizer
    loss_func = torch.nn.MSELoss()  # For regression tasks like this, MSE is a good choice
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    metrics = {"train_longitudinal_error": [], "train_lateral_error": [], "val_longitudinal_error": [], "val_lateral_error": []}

    # Training loop
    for epoch in range(num_epoch):
        # Clear metrics at the beginning of the epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        for track_left, track_right, waypoints, mask in train_data:
            track_left, track_right, waypoints, mask = track_left.to(device), track_right.to(device), waypoints.to(device), mask.to(device)

            # Training step
            optimizer.zero_grad()
            pred_waypoints = model(track_left, track_right)  # Get predicted waypoints
            loss = loss_func(pred_waypoints, waypoints)  # Compute MSE loss
            loss.backward()
            optimizer.step()

            # Compute errors
            longitudinal_error, lateral_error = compute_errors(pred_waypoints, waypoints)
            metrics["train_longitudinal_error"].append(longitudinal_error)
            metrics["train_lateral_error"].append(lateral_error)

        # Disable gradient computation and switch to evaluation mode for validation
        with torch.no_grad():
            model.eval()

            for track_left, track_right, waypoints, mask in val_data:
                track_left, track_right, waypoints, mask = track_left.to(device), track_right.to(device), waypoints.to(device), mask.to(device)

                # Validation step
                pred_waypoints = model(track_left, track_right)
                longitudinal_error, lateral_error = compute_errors(pred_waypoints, waypoints)
                metrics["val_longitudinal_error"].append(longitudinal_error)
                metrics["val_lateral_error"].append(lateral_error)

        # Log average train and val errors to TensorBoard
        epoch_train_longitudinal_error = torch.as_tensor(metrics["train_longitudinal_error"]).mean()
        epoch_train_lateral_error = torch.as_tensor(metrics["train_lateral_error"]).mean()
        epoch_val_longitudinal_error = torch.as_tensor(metrics["val_longitudinal_error"]).mean()
        epoch_val_lateral_error = torch.as_tensor(metrics["val_lateral_error"]).mean()

        logger.add_scalar("train/longitudinal_error", epoch_train_longitudinal_error, global_step)
        logger.add_scalar("train/lateral_error", epoch_train_lateral_error, global_step)
        logger.add_scalar("val/longitudinal_error", epoch_val_longitudinal_error, global_step)
        logger.add_scalar("val/lateral_error", epoch_val_lateral_error, global_step)

        # Print stats on the first, last, and every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_longitudinal_error={epoch_train_longitudinal_error:.4f} "
                f"train_lateral_error={epoch_train_lateral_error:.4f} "
                f"val_longitudinal_error={epoch_val_longitudinal_error:.4f} "
                f"val_lateral_error={epoch_val_lateral_error:.4f}"
            )

        global_step += 1

    # Save the model for grading
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
