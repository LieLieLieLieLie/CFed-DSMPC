import torch
import torch.nn as nn
from config import Config


class DensityEstimator(nn.Module):
    """
    Autoencoder that learns the distribution of (state, action) pairs
    observed during embodied exploration.

    At inference:  shift = ||x - decoder(encoder(x))||_2
    Large shift  → (s, a) is out-of-distribution (OOD) for this agent cluster.
    Small shift  → (s, a) was frequently encountered → agent can act confidently.
    """

    def __init__(self):
        super().__init__()
        # input dim: state(4) + action(2) = 6
        self.encoder = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 6),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: (B, 4) state  [x, y, v, yaw]
            a: (B, 2) action [steer, accel]
        Returns:
            x_hat: (B, 6) reconstructed input
        """
        x = torch.cat([s, a], dim=-1)          # (B, 6)
        return self.decoder(self.encoder(x))    # (B, 6)

    def compute_shift(self, s_np, a_np) -> float:
        """
        Compute reconstruction error (distribution shift) for a single
        (state, action) pair given as numpy arrays.
        Returns a non-negative scalar; higher = more OOD.
        """
        self.eval()
        with torch.no_grad():
            s_t = torch.FloatTensor(s_np).unsqueeze(0).to(Config.DEVICE)  # (1,4)
            a_t = torch.FloatTensor(a_np).unsqueeze(0).to(Config.DEVICE)  # (1,2)
            x   = torch.cat([s_t, a_t], dim=-1)                           # (1,6)
            x_hat = self.decoder(self.encoder(x))
            shift = torch.norm(x - x_hat, p=2, dim=-1).item()
        return shift
