import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.utils.tensorboard as tb
from .models import MLPPlanner, TransformerPlanner, CNNPlanner, load_model, save_model
from .datasets.road_dataset import load_data

# Custom loss function based on longitudinal and lateral error
def compute_errors(predictions, targets):
    # Assuming predictions and targets are tensors of shape (batch_size, num_waypoints, 2) where each waypoint is (x, y)
    pred_x, pred_y = predictions[..., 0], predictions[..., 1]
    target_x, target_y = targets[..., 0], targets[..., 1]

    # Longitudinal error is the absolute difference in the x-direction
    longitudinal_error = torch.mean(torch.abs(pred_x - target_x))

    # Lateral error is the absolute difference in the y-direction
    lateral_error = torch.mean(torch.abs(pred_y - target_y))

    return longitudinal_error, lateral_error

def train(
    model_name: str = "mlp_planner.th",
    transform_pipeline="state_only",
    num_workers: int = 2,
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 2024,
    exp_dir: str = "logs",
    **kwargs,
):
    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Load model
    model = load_model(model_name, **kwargs).to(device)
    model.train()

    # Load datasets
    train_loader = load_data("drive_data/train", transform_pipeline=transform_pipeline, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    val_loader = load_data("drive_data/val", transform_pipeline=transform_pipeline, shuffle=False, batch_size=batch_size, num_workers=0)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    global_step = 0

    for epoch in range(num_epoch):
        model.train()
        train_longitudinal_errors, train_lateral_errors = [], []

        for batch in train_loader:
            optimizer.zero_grad()

            input_data, targets = process_input(batch, model_name, device)
            preds = model(*input_data)

            # Compute errors
            longitudinal_error, lateral_error = compute_errors(preds, targets)

            # Loss: A combination of longitudinal and lateral error
            total_loss = longitudinal_error + lateral_error
            total_loss.backward()
            optimizer.step()

            # Log errors
            train_longitudinal_errors.append(longitudinal_error.item())
            train_lateral_errors.append(lateral_error.item())
            global_step += 1

        avg_train_longitudinal_error = np.mean(train_longitudinal_errors)
        avg_train_lateral_error = np.mean(train_lateral_errors)

        # Validation
        model.eval()
        val_longitudinal_errors, val_lateral_errors = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_data, targets = process_input(batch, model_name, device)
                preds = model(*input_data)

                longitudinal_error, lateral_error = compute_errors(preds, targets)
                val_longitudinal_errors.append(longitudinal_error.item())
                val_lateral_errors.append(lateral_error.item())

        avg_val_longitudinal_error = np.mean(val_longitudinal_errors)
        avg_val_lateral_error = np.mean(val_lateral_errors)

        # Log to TensorBoard
        logger.add_scalar("train_longitudinal_error", avg_train_longitudinal_error, global_step)
        logger.add_scalar("train_lateral_error", avg_train_lateral_error, global_step)
        logger.add_scalar("val_longitudinal_error", avg_val_longitudinal_error, global_step)
        logger.add_scalar("val_lateral_error", avg_val_lateral_error, global_step)

        print(f"Epoch {epoch+1:02d}/{num_epoch} "
              f"Train Longitudinal Error: {avg_train_longitudinal_error:.4f} | "
              f"Train Lateral Error: {avg_train_lateral_error:.4f} | "
              f"Val Longitudinal Error: {avg_val_longitudinal_error:.4f} | "
              f"Val Lateral Error: {avg_val_lateral_error:.4f}")

    # Save model
    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


def process_input(batch, model_name, device):
    if model_name == "cnn_planner":
        images = batch["image"].to(device)
        targets = batch["waypoints"].to(device)
        return images, targets
    else:
        track_left = batch["track_left"].to(device)
        track_right = batch["track_right"].to(device)
        targets = batch["waypoints"].to(device)
        return (track_left, track_right), targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--transform_pipeline", type=str, default="state_only")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--exp_dir", type=str, default="logs")

    train(**vars(parser.parse_args()))
