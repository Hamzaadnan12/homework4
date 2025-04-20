import torch
import csv
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# For computing errors
def compute_errors(pred_waypoints, true_waypoints):
    """
    Compute longitudinal and lateral errors between predicted and true waypoints.
    Args:
        pred_waypoints (torch.Tensor): The predicted waypoints (B, n_waypoints, 2).
        true_waypoints (torch.Tensor): The ground truth waypoints (B, n_waypoints, 2).
    Returns:
        tuple: Longitudinal error, Lateral error.
    """
    longitudinal_error = torch.abs(pred_waypoints[:, :, 0] - true_waypoints[:, :, 0]).mean()
    lateral_error = torch.abs(pred_waypoints[:, :, 1] - true_waypoints[:, :, 1]).mean()

    return longitudinal_error.item(), lateral_error.item()

# Saving and Loading Models
def save_model(model, filename):
    """
    Save the model's state dict to a file.
    Args:
        model (torch.nn.Module): The PyTorch model to save.
        filename (str): The filename to save the model as.
    """
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


def load_model(model, filename):
    """
    Load the model's state dict from a file.
    Args:
        model (torch.nn.Module): The PyTorch model to load the state dict into.
        filename (str): The filename to load the model from.
    """
    model.load_state_dict(torch.load(filename))
    print(f"Model loaded from {filename}")
    model.eval()  # Set the model to evaluation mode

    return model


# Custom Dataset for SuperTuxKart (same as previous, but adjusted for road dataset)
class RoadDataset(Dataset):
    def __init__(self, dataset_path: str, split: str = 'train'):
        """
        This is a custom dataset for SuperTuxKart lane data.
        Arguments:
            dataset_path (str): Path to the dataset folder.
            split (str): Either 'train' or 'test' (default is 'train').
        """
        self.dataset_path = Path(dataset_path)
        self.split = split

        self.data = []
        self.load_data()

    def load_data(self):
        """
        Load and parse the road data (lane boundary and waypoints).
        """
        # Example: Let's assume CSV files or other data format
        with open(self.dataset_path / f"{self.split}_data.csv", newline="") as f:
            for row in csv.reader(f):
                # Assuming CSV format: [track_left, track_right, waypoints, waypoints_mask]
                track_left = torch.tensor([float(x) for x in row[0].split()], dtype=torch.float32)
                track_right = torch.tensor([float(x) for x in row[1].split()], dtype=torch.float32)
                waypoints = torch.tensor([float(x) for x in row[2].split()], dtype=torch.float32)
                mask = torch.tensor([int(x) for x in row[3].split()], dtype=torch.bool)

                self.data.append((track_left, track_right, waypoints, mask))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        track_left, track_right, waypoints, mask = self.data[idx]
        return {"track_left": track_left, "track_right": track_right, "waypoints": waypoints, "waypoints_mask": mask}


def load_road_data(dataset_path: str, split: str = 'train', num_workers: int = 0, batch_size: int = 64, shuffle: bool = False) -> DataLoader:
    """
    Loads the road dataset and returns a DataLoader.
    Arguments:
        dataset_path: Path to the dataset folder.
        split: 'train' or 'test'.
        num_workers: Number of workers for data loading.
        batch_size: Batch size for DataLoader.
        shuffle: Whether to shuffle the data.
    """
    dataset = RoadDataset(dataset_path, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
