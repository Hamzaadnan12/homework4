import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.utils.tensorboard as tb
from models import MLPPlanner, TransformerPlanner, CNNPlanner, load_model, save_model
from datasets.road_dataset import load_data


# Custom loss function based on longitudinal and lateral error
def compute_errors(predictions, targets):
    """
    Compute longitudinal and lateral errors.

    Args:
        predictions (torch.Tensor): predicted waypoints of shape (batch_size, num_waypoints, 2)
        targets (torch.Tensor): ground truth waypoints of shape (batch_size, num_waypoints, 2)

    Returns:
        longitudinal_error (torch.Tensor): the mean absolute error in the x-direction (longitudinal)
        lateral_error (torch.Tensor): the mean absolute error in the y-direction (lateral)
    """
    pred_x, pred_y = predictions[..., 0], predictions[..., 1]
    target_x, target_y = targets[..., 0], targets[..., 1]

    # Longitudinal error is the absolute difference in the x-direction
    longitudinal_error = torch.mean(torch.abs(pred_x - target_x))

    # Lateral error is the absolute difference in the y-direction
    lateral_error = torch.mean(torch.abs(pred_y - target_y))

    return longitudinal_error, lateral_error


# Main training function
def train(
    model_name: str = "mlp_planner",
    transform_pipeline="state_only",
    num_workers: int = 2,
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 2024,
    exp_dir: str = "logs",
    **kwargs,
):
    """
    Main training loop for training various planners (MLP, Transformer, CNN).

    Args:
        model_name (str): Name of the model to train (mlp_planner, transformer_planner, cnn_planner)
        transform_pipeline (str): The transform pipeline used for preprocessing
        num_workers (int): Number of data loader workers
        num_epoch (int): Number of training epochs
        lr (float): Learning rate for the optimizer
        batch_size (int): Batch size for training
        seed (int): Random seed for reproducibility
        exp_dir (str): Directory to save training logs and models
        **kwargs: Additional keyword arguments to pass to the model
    """
    # Setup random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup TensorBoard logging
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Load the model
    model = load_model(model_name, **kwargs).to(device)
    model.train()

    # Load datasets
    train_loader = load_data("drive_data/train", transform_pipeline=transform_pipeline, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    val_loader = load_data("drive_data/val", transform_pipeline=transform_pipeline, shuffle=False, batch_size=batch_size, num_workers=0)

    # Set up the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    global_step = 0

    for epoch in range(num_epoch):
        model.train()
        train_longitudinal_errors, train_lateral_errors = [], []

        # Train loop
        for batch in train_loader:
            optimizer.zero_grad()

            input_data, targets = process_input(batch, model_name, device)
            preds = model(*input_data)

            # Compute longitudinal and lateral errors
            longitudinal_error, lateral_error = compute_errors(preds, targets)

            # Total loss: combination of longitudinal and lateral error
            total_loss = longitudinal_error + lateral_error
            total_loss.backward()
            optimizer.step()

            # Log the errors
            train_longitudinal_errors.append(longitudinal_error.item())
            train_lateral_errors.append(lateral_error.item())
            global_step += 1

        # Average training errors
        avg_train_longitudinal_error = np.mean(train_longitudinal_errors)
        avg_train_lateral_error = np.mean(train_lateral_errors)

        # Validation loop
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

        # Print results
        print(f"Epoch {epoch+1:02d}/{num_epoch} "
              f"Train Longitudinal Error: {avg_train_longitudinal_error:.4f} | "
              f"Train Lateral Error: {avg_train_lateral_error:.4f} | "
              f"Val Longitudinal Error: {avg_val_longitudinal_error:.4f} | "
              f"Val Lateral Error: {avg_val_lateral_error:.4f}")

    # Save the trained model
    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


# Helper function to process input data based on the model type
def process_input(batch, model_name, device):
    """
    Processes the input data depending on the model type (CNN, MLP, Transformer).

    Args:
        batch (dict): The input batch of data
        model_name (str): The model name to determine input processing
        device (torch.device): The device to transfer tensors to

    Returns:
        tuple: The processed input data and target data
    """
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

    # Define command line arguments
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--transform_pipeline", type=str, default="state_only")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--exp_dir", type=str, default="logs")

    # Start training
    train(**vars(parser.parse_args()))
