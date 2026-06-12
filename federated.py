"""
federated.py  —  Federated learning components

FedClient  : local DensityEstimator + FedProx training
FedServer  : clustered FedAvg (one global model per cluster)  ← proposed
FedAvgServer: naïve FedAvg (one global model for ALL agents)  ← baseline-3
"""

import torch
import torch.optim as optim
import numpy as np
from config import Config
from models import DensityEstimator


# ─────────────────────────────────────────────────────────────────────────────
#  Federated Client  (shared by both server types)
# ─────────────────────────────────────────────────────────────────────────────
class FedClient:
    def __init__(self, client_id: int, cluster_id: int):
        self.id         = client_id
        self.cluster_id = cluster_id
        self.den_model  = DensityEstimator().to(Config.DEVICE)
        self.data_buffer: list[tuple] = []

    def add_data(self, s: np.ndarray, a: np.ndarray):
        self.data_buffer.append((s.copy(), a.copy()))

    def train(self, global_state_dict: dict):
        if len(self.data_buffer) < 5:
            return
        self.den_model.load_state_dict(global_state_dict)
        opt = optim.Adam(self.den_model.parameters(), lr=Config.LR)
        mse = torch.nn.MSELoss()
        global_params = [p.clone().detach() for p in self.den_model.parameters()]

        s_arr = np.array([item[0] for item in self.data_buffer], dtype=np.float32)
        a_arr = np.array([item[1] for item in self.data_buffer], dtype=np.float32)
        s_t   = torch.FloatTensor(s_arr).to(Config.DEVICE)
        a_t   = torch.FloatTensor(a_arr).to(Config.DEVICE)
        x_target = torch.cat([s_t, a_t], dim=-1)

        for _ in range(Config.LOCAL_EPOCHS):
            opt.zero_grad()
            x_hat   = self.den_model(s_t, a_t)
            l_recon = mse(x_hat, x_target)
            l_prox  = sum(
                (w - wg).pow(2).sum()
                for w, wg in zip(self.den_model.parameters(), global_params)
            )
            loss = l_recon + (Config.PROXIMAL_MU / 2.0) * l_prox
            loss.backward()
            opt.step()


# ─────────────────────────────────────────────────────────────────────────────
#  Clustered FedAvg Server  (proposed — one model per cluster)
# ─────────────────────────────────────────────────────────────────────────────
class FedServer:
    """One global DensityEstimator per cluster; aggregation within cluster only."""

    def __init__(self):
        self.global_den: dict[int, DensityEstimator] = {
            k: DensityEstimator().to(Config.DEVICE)
            for k in range(Config.NUM_CLUSTERS)
        }
        self.collected_states: dict[int, list] = {
            k: [] for k in range(Config.NUM_CLUSTERS)
        }

    def aggregate(self, clients: list[FedClient], cluster_id: int = 0):
        target_clients = [c for c in clients if c.cluster_id == cluster_id]
        if not target_clients:
            return
        if len(self.collected_states[cluster_id]) == 0:
            for c in target_clients:
                self.collected_states[cluster_id].extend(
                    [item[0][:2].tolist() for item in c.data_buffer]
                )
        global_sd = self.global_den[cluster_id].state_dict()
        for key in global_sd.keys():
            stacked = torch.stack(
                [c.den_model.state_dict()[key].float() for c in target_clients]
            )
            global_sd[key] = stacked.mean(dim=0)
        self.global_den[cluster_id].load_state_dict(global_sd)


# ─────────────────────────────────────────────────────────────────────────────
#  Naïve FedAvg Server  (baseline-3 — single shared model, no clustering)
# ─────────────────────────────────────────────────────────────────────────────
class FedAvgServer:
    """
    Single global DensityEstimator trained on ALL agents' data pooled together.
    Implements McMahan et al. (AISTATS 2017) without any clustering.
    Used by FedAvgMPC as the shared density prior.
    """

    def __init__(self):
        self.global_den = DensityEstimator().to(Config.DEVICE)
        # Mirror interface of FedServer for plotting convenience
        self.collected_states: dict[int, list] = {
            k: [] for k in range(Config.NUM_CLUSTERS)
        }

    def aggregate(self, clients: list[FedClient]):
        """FedAvg across ALL clients regardless of cluster."""
        if not clients:
            return
        # Collect states once for visualisation (split by cluster for compat.)
        for c in clients:
            if len(self.collected_states[c.cluster_id]) == 0:
                self.collected_states[c.cluster_id].extend(
                    [item[0][:2].tolist() for item in c.data_buffer]
                )
        global_sd = self.global_den.state_dict()
        for key in global_sd.keys():
            stacked = torch.stack(
                [c.den_model.state_dict()[key].float() for c in clients]
            )
            global_sd[key] = stacked.mean(dim=0)
        self.global_den.load_state_dict(global_sd)
