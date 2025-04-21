from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        input_dim = n_track * 4  # We concatenate left and right, so input size is n_track * 2
        
        hidden_dim = 128
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Flatten input (left + right)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_waypoints * 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Flatten the track data (combine left and right track boundaries)
        x = torch.cat([track_left, track_right], dim=-1)  # Shape (b, n_track, 4)
        
        # Fix applied: Flatten to (b, n_track * 2) instead of (b, n_track * 4)
        x = x.view(x.size(0), -1)  # Flatten to (b, n_track * 2)  <-- THIS IS THE FIX!

        # Pass through the MLP layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        # Reshape to (b, n_waypoints, 2)
        x = x.view(x.size(0), self.n_waypoints, 2)

        return x



class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.query_embed = nn.Embedding(n_waypoints, d_model)
  
        nhead = 4
        num_layers = 4
        # Transformer Encoder (takes in the lane boundaries as memory)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Fully connected layers for final output
        self.fc = nn.Linear(d_model, 2)  # Predict (x, y) for each waypoint

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Combine left and right boundaries (shape: b, n_track, 4)
        track_data = torch.cat([track_left, track_right], dim=-1)

        # Reshape track data to (b, n_track, d_model) for transformer
        track_data = track_data.view(track_data.size(0), self.n_track, 2 * self.d_model)

        # Pass the track data through the transformer encoder
        memory = self.transformer_encoder(track_data)

        # Create waypoint query embeddings
        query = self.query_embed.weight.unsqueeze(0).repeat(track_data.size(0), 1, 1)

        # Apply transformer decoder to generate waypoints
        # Here, we use the query embeddings and memory from the encoder
        transformer_decoder_output = self.fc(memory)

        # Reshape the output to (b, n_waypoints, 2)
        waypoints = transformer_decoder_output.view(track_data.size(0), self.n_waypoints, 2)

        return waypoints


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()
        input_channels = 3 # RGB image
        input_height = 96   # Image height (96)
        input_width = 128   #

        self.n_waypoints = n_waypoints

        self.n_waypoints = n_waypoints

        # Define the CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),  # (B, 3, 96, 128) -> (B, 32, 48, 64)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (B, 32, 48, 64) -> (B, 64, 24, 32)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (B, 64, 24, 32) -> (B, 128, 12, 16)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (B, 128, 12, 16) -> (B, 256, 6, 8)
            nn.ReLU(),
            nn.Flatten(),
        )

        # Fully connected layer to output waypoints (n_waypoints, 2)
        # Output shape should be (B, n_waypoints * 2)
        self.fc = nn.Linear(256 * 6 * 8, n_waypoints * 2)

        # Normalize the input using pre-calculated means and std devs
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)
        

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        # Normalize input image
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Pass through CNN backbone
        x = self.backbone(x)

        # Pass through fully connected layer to get waypoints
        x = self.fc(x)

        # Reshape to (B, n_waypoints, 2) for output
        x = x.view(x.size(0), self.n_waypoints, 2)

        return x


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
