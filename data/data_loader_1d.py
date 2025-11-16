import torch
import os
import re
from torch.utils.data import Dataset
from pathlib import Path


class BurgersDataset(Dataset):
    def __init__(self, data, dt=None, mstep=1, starts_gap=None):
        """
        Args:
            data (dict): data dictionary containing simulation trajectories and parameters.
            dt (float, optional): The desired time interval between frames used in the target trajectory.
                                  Must be an integer multiple of the generation dt.
                                  If None, the dataset uses the generation dt.
            mstep (int): The number of steps forward (with dt) to include in the target trajectory.
                         For example, mstep=5 returns 5 frames (each dt apart) as the target.
            starts_gap (int): The gap between starting indices for the target trajectory.
                              For example, starts_gap=2 returns every other starting index.
                              It can be None, then it will be set to dt/gen_dt.
        """
        self.starts_gap = starts_gap
        self.trajectories = data["trajectories"]  # shape: [B, T+1, D]
        self.nu = data["nu"]                      # shape: [B, 1]
        self.A = data["A"]                        # shape: [B, 1, J]
        self.omega = data["omega"]                # shape: [B, 1, J]
        self.phi = data["phi"]                    # shape: [B, 1, J]
        self.l = data["l"]                        # shape: [B, 1, J]
        self.metadata = data["metadata"]          # metadata containing generation parameters
        self.gen_dt = self.metadata["gen_dt"]     # Time step used for generation

        # Validate dt and compute substeps
        self.dt = dt if dt is not None else self.gen_dt
        ratio = self.dt / self.gen_dt
        self.starts_gap=ratio if self.starts_gap is None else self.starts_gap
        if abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(f"dt must be a multiple of gen_dt ({self.gen_dt})")
        self.n_substeps = int(round(ratio))
        
        # Calculate valid trajectory indices
        total_steps = self.trajectories.shape[1] - 1 # minus 1 for initial state
        max_start = total_steps - self.n_substeps * mstep # max starting index
        self.starts = torch.arange(0, max_start + 1, self.starts_gap) # valid starting indices
        self.mstep = mstep

    def __len__(self):
        return self.trajectories.shape[0] * len(self.starts) # B * num_starts, if n_substeps=1, B * (T-mstep+1), add 1 for initial state, so 1 steps as intial state, mstep steps forward. 

    def __getitem__(self, idx):
        batch_idx = idx // len(self.starts)
        start_idx = idx % len(self.starts)
        t_idx = self.starts[start_idx]
        
        # Extract trajectory segment
        end_idx = t_idx + self.n_substeps * self.mstep
        trajectory = self.trajectories[batch_idx, t_idx:end_idx+1:self.n_substeps]
        
        return {
            "u_initial": trajectory[0],            # [D]
            "u_target": trajectory[1:],            # [mstep, D]
            "nu": self.nu[batch_idx],              # [1]
            "A": self.A[batch_idx],                # [1, J]
            "omega": self.omega[batch_idx],
            "phi": self.phi[batch_idx],
            "l": self.l[batch_idx],
            "t": t_idx * self.gen_dt,              # original simulation time for u_initial
        }
    
class KSDataset(Dataset):
    def __init__(self, data, dt=None, mstep=1,starts_gap=None):
        """
        Args:
            data (dict): data dictionary containing simulation trajectories and parameters.
            dt (float, optional): The desired time interval between frames used in the target trajectory.
                                  Must be an integer multiple of the generation dt.
                                  If None, the dataset uses the generation dt.
            mstep (int): The number of steps forward (with dt) to include in the target trajectory.
                         For example, mstep=5 returns 5 frames (each dt apart) as the target.
        """
        self.starts_gap = starts_gap
        self.trajectories = data["trajectories"]  # shape: [B, T+1, D]
        self.domain_size = data["domain_size"]    # shape: [B, 1]
        self.metadata = data["metadata"]          # metadata containing generation parameters
        self.gen_dt = self.metadata["gen_dt"]     # Time step used for generation

        
        # Validate dt and compute substeps
        self.dt = dt if dt is not None else self.gen_dt
        ratio = self.dt / self.gen_dt
        self.starts_gap=ratio if self.starts_gap is None else self.starts_gap

        if abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(f"dt must be a multiple of gen_dt ({self.gen_dt})")
        self.n_substeps = int(round(ratio))
        
        # Calculate valid trajectory indices
        total_steps = self.trajectories.shape[1] - 1 # minus 1 for initial state
        max_start = total_steps - self.n_substeps * mstep # max starting index
        self.starts = torch.arange(0, max_start + 1, self.starts_gap) # valid starting indices
        self.mstep = mstep

    def __len__(self):
        return self.trajectories.shape[0] * len(self.starts) # B * num_starts, if n_substeps=1, B * (T-mstep+1), add 1 for initial state, so 1 steps as intial state, mstep steps forward. 

    def __getitem__(self, idx):
        batch_idx = idx // len(self.starts)
        start_idx = idx % len(self.starts)
        t_idx = self.starts[start_idx]
        
        # Extract trajectory segment
        end_idx = t_idx + self.n_substeps * self.mstep
        trajectory = self.trajectories[batch_idx, t_idx:end_idx+1:self.n_substeps]
        
        return {
            "u_initial": trajectory[0],            # [D]
            "u_target": trajectory[1:],            # [mstep, D]
            "domain_size": self.domain_size[batch_idx],
        }
    
def BG_collate_fn(batch):
    collated = {
        "u_initial": torch.stack([item["u_initial"] for item in batch]),   # [B, D]
        "u_target": torch.stack([item["u_target"] for item in batch]),     # [B, mstep, D]
        "nu": torch.stack([item["nu"] for item in batch]),                 # [B, 1]
        "A": torch.stack([item["A"] for item in batch]),                   # [B, 1, J]
        "omega": torch.stack([item["omega"] for item in batch]),           # [B, 1, J]
        "phi": torch.stack([item["phi"] for item in batch]),               # [B, 1, J]
        "l": torch.stack([item["l"] for item in batch]),                   # [B, 1, J]
        "t": torch.tensor([item["t"] for item in batch]),                  # [B]
    }
    return collated

def KS_collate_fn(batch):
    collated = {
        "u_initial": torch.stack([item["u_initial"] for item in batch]),   # [B, D]
        "u_target": torch.stack([item["u_target"] for item in batch]),     # [B, mstep, D]
        "domain_size": torch.tensor([item["domain_size"] for item in batch]),                  # [B,1]
    }
    return collated