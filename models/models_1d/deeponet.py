import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_grid



class BranchNet(nn.Module):
    """Processes input function samples at fixed sensors"""
    def __init__(self, sensor_dim, hidden_dim, output_dim, num_layers=4):
        super().__init__()
        layers = [nn.Linear(sensor_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers-2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # [B, output_dim]

class TrunkNet(nn.Module):
    """Processes evaluation coordinates"""
    def __init__(self, coord_dim, hidden_dim, output_dim, num_layers=4):
        super().__init__()
        layers = [nn.Linear(coord_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers-2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # [B*num_points, output_dim]
class BranchNetRNN(nn.Module):
    """Process temporal sequences with GRU"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [B, T_in, sensor_dim]
        _, h_n = self.gru(x)  # h_n: [num_layers, B, hidden_dim]
        return self.fc(h_n[-1])  # [B, output_dim]
class DeepONet1D_RNN(nn.Module):
    def __init__(self, L=16, sensor_dim=100, T_in=10, coord_dim=2, 
                 hidden=128, output_dim=128, trunk_layers=4, branch_layers=4):
        super().__init__()
        self.L = L
        self.T_in = T_in
        self.periodic_terms = 4
        coord_dim += self.periodic_terms *2 # 2+2*n,n: Fourier features, 2:x+1 (x for physical location, 1 for Fourier features)
        # Modification 1: Expand branch input to handle T_in time steps
        self.branch = BranchNet(sensor_dim * T_in, hidden, output_dim, branch_layers)
        
        # Modification 2: Add time coordinate to trunk
        self.trunk = TrunkNet(coord_dim, hidden, output_dim, trunk_layers)
        self.bias = nn.Parameter(torch.zeros(1))
        self._input_transform = self.periodic
    
    @staticmethod
    def periodic(x,n):
        x = x * 2 * torch.pi
        features = [x, torch.ones_like(x)]
        for i in range(1, n+1):
            features.extend([torch.cos(i*x), torch.sin(i*x)])
        return torch.cat(features, dim=-1)
    
    def forward(self, u, L=None):
        """
        Args:
            u: [B, T_in, sensor_dim] input function values
        Returns:
            [B, 1, sensor_dim] prediction for next timestep
        """
        # Reshape u: [B, sensor_dim*T_in] (like FNO's channel expansion)
        L = L if L is not None else self.L
        u = u.permute(0, 2, 1)  # [B, T, D] -> [B, D, T]

        B, D, T = u.shape
        u_flat = u.reshape(B, -1)
        
        # Branch processes flattened temporal sequence
        branch_out = self.branch(u_flat)  # [B, output_dim]
        
        grid = get_grid(u, L)  # Spatial coordinates [B, D, 1]
        if self._input_transform is not None:
            grid = self._input_transform(grid,n=self.periodic_terms)  # [B, D, coord_dim]
        # Trunk processes space-time
        trunk_flat = grid.view(-1, grid.shape[-1])
        trunk_out = self.trunk(trunk_flat).view(B, D, -1)
        # trunk_out = self.trunk(grid.view(-1, 2)).view(B, D, -1)  # [B, D, output_dim]
        output = torch.einsum('bi,bpi->bp', branch_out, trunk_out) + self.bias # [B, D]
        return output.unsqueeze(1) # [B, 1, D]
    
class DeepONet1D(nn.Module):
    def __init__(self, L=16, sensor_dim=100, coord_dim=2, 
                 hidden=128, output_dim=128, trunk_layers=4, branch_layers=4):
        super().__init__()
        self.L = L
        self.periodic_terms = 4
        coord_dim += self.periodic_terms *2 
        self.branch = BranchNet(sensor_dim, hidden, output_dim, branch_layers)
        self.trunk = TrunkNet(coord_dim, hidden, output_dim, trunk_layers)
        self.bias = nn.Parameter(torch.zeros(1))  # Learnable scalar bias
        self._input_transform = self.periodic

    @staticmethod
    def periodic(x,n):
        x = x * 2 * torch.pi
        features = [x, torch.ones_like(x)]
        for i in range(1, n+1):
            features.extend([torch.cos(i*x), torch.sin(i*x)])
        return torch.cat(features, dim=-1)
    
    def apply_feature_transform(self, transform):
        """Apply a transform to the trunk net inputs (features)."""
        self._input_transform = transform

    
    def forward(self, u, L=None):
        """
        Args:
            u: Input function values [B, sensor_dim]
        Returns:
            [B, num_points] predictions at locations y
        """
        L = L if L is not None else self.L
        # Branch: [B, sensor_dim] -> [B, output_dim]
        branch_out = self.branch(u) 
        y = get_grid(u,L)  # [B, num_points, coord_dim]
        # Apply feature transform to trunk inputs (Fourier features)
        if self._input_transform is not None:
            y = self._input_transform(y,n=self.periodic_terms)
        # Trunk: [B, num_points, coord_dim] -> [B, num_points, output_dim]
        B, num_points, _ = y.shape
        trunk_flat = y.view(-1, y.shape[-1])  # [B*num_points, coord_dim]
        trunk_out = self.trunk(trunk_flat).view(B, num_points, -1)
        
        # Dot product + bias
        return torch.einsum('bi,bpi->bp', branch_out, trunk_out) + self.bias
    
class DeepONet1D_GRU(nn.Module):
    def __init__(self, L=16, sensor_dim=100, T_in=10, coord_dim=3, 
                 hidden=128, output_dim=128, trunk_layers=4, branch_layers=2):
        super().__init__()
        self.L = L
        self.T_in = T_in
        self.periodic_terms = 4
        coord_dim += self.periodic_terms *2 # 2+2*n,n: Fourier features, 3:x+1+1 (x for physical location, 1 for Fourier features, 1 for time)
        # Modification 1: Expand branch input to handle T_in time steps
        self.branch = BranchNetRNN(sensor_dim, hidden, output_dim, branch_layers)
        
        # Modification 2: Add time coordinate to trunk
        self.trunk = TrunkNet(coord_dim, hidden, output_dim, trunk_layers)
        self.bias = nn.Parameter(torch.zeros(1))
        self._input_transform = self.periodic
    
    @staticmethod
    def periodic(x,n):
        features = [x, torch.ones_like(x)]
        x = x * 2 * torch.pi
        for i in range(1, n+1):
            features.extend([torch.cos(i*x), torch.sin(i*x)])
        return torch.cat(features, dim=-1)
        
    
    def forward(self, u, L=None):
        """
        Args:
            u: [B, T_in, sensor_dim] input function values
        Returns:
            [B, 1, sensor_dim] prediction for next timestep
        """
        # Reshape u: [B, sensor_dim*T_in] (like FNO's channel expansion)
        L = L if L is not None else self.L
        # u = u.permute(0, 2, 1)  # [B, T, D] -> [B, D, T]

        B, T, D = u.shape
        # u_flat = u.reshape(B, -1)
        
        # Branch processes flattened temporal sequence
        branch_out = self.branch(u)  # [B, output_dim]
        
        grid = get_grid(u.permute(0, 2, 1), L)  # Spatial coordinates [B, D, 1]
        target_time = torch.ones(B, D, 1).to(u.device) * (self.T_in + 1)/self.T_in  # [B, D, 1]
        
        if self._input_transform is not None:
            grid = self._input_transform(grid,n=self.periodic_terms)  # [B, D, coord_dim-1]
        # Trunk processes space-time
        grid_time = torch.cat([grid, target_time], dim=-1)  # [B, D, coord_dim]
        trunk_flat = grid_time.view(-1, grid_time.shape[-1])
        trunk_out = self.trunk(trunk_flat).view(B, D, -1)
        # trunk_out = self.trunk(grid.view(-1, 2)).view(B, D, -1)  # [B, D, output_dim]
        output = torch.einsum('bi,bpi->bp', branch_out, trunk_out) + self.bias # [B, D]
        return output.unsqueeze(1) # [B, 1, D]