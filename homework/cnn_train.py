import os
import torch
import torch.nn as nn
from tqdm import tqdm

from models import CNNPlanner
from datasets.road_dataset import load_data

# --- Custom Loss ---
def masked_mse(predictions, targets, masks):
    squared_error = (predictions - targets) ** 2
    masked_loss = squared_error.sum(dim=2) * masks.float()
    return masked_loss.sum() / (masks.sum() + 1e-6)

# --- Metric Calculation ---
def evaluate_errors(predictions, targets):
    delta = predictions - targets
    long_error = delta[..., 1].abs().mean().item()
    lat_error = delta[..., 0].abs().mean().item()
    return long_error, lat_error

# --- Main Training Routine ---
def run_training(
    data_dir=None,
    num_epochs=80,
    batch_sz=128,
    learning_rate=5e-4,
    model_out="cnn_planner.th",
):
    # Default path resolution
    if data_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(current_dir, "../drive_data"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset...")
    train_data = load_data(
        dataset_path=os.path.join(data_dir, "train"),
        transform_pipeline="default",
        return_dataloader=True,
        batch_size=batch_sz,
        shuffle=True
    )

    val_data = load_data(
        dataset_path=os.path.join(data_dir, "val"),
        transform_pipeline="default",
        return_dataloader=True,
        batch_size=batch_sz,
        shuffle=False
    )

    print(f"Training batches: {len(train_data)} | Validation batches: {len(val_data)}")

    print("Initializing model...")
    net = CNNPlanner(n_waypoints=3).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        net.train()
        epoch_loss = 0.0

        for sample in tqdm(train_data, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs = sample["image"].to(device)
            gt_points = sample["waypoints"].to(device)
            masks = sample["waypoints_mask"].to(device)

            preds = net(imgs)
            loss = masked_mse(preds, gt_points, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_data)
        print(f"[Epoch {epoch+1}] Training Loss: {avg_loss:.4f}")

        # --- Evaluate on validation set ---
        net.eval()
        total_long, total_lat, val_batches = 0.0, 0.0, 0

        with torch.no_grad():
            for sample in val_data:
                imgs = sample["image"].to(device)
                gt_points = sample["waypoints"].to(device)
                masks = sample["waypoints_mask"].to(device)

                preds = net(imgs)
                pred_valid = preds[masks]
                gt_valid = gt_points[masks]

                if pred_valid.numel() > 0:
                    long_err, lat_err = evaluate_errors(pred_valid, gt_valid)
                    total_long += long_err
                    total_lat += lat_err
                    val_batches += 1

        if val_batches > 0:
            print(f"Validation | Longitudinal Error: {total_long / val_batches:.4f} | Lateral Error: {total_lat / val_batches:.4f}")

    torch.save(net.state_dict(), model_out)
    print(f"Model saved to {model_out}")

# --- Entry Point ---
if __name__ == "__main__":
    run_training()
