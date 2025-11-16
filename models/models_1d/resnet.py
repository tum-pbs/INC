import torch
import torch.nn as nn
from .utils import get_grid

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2, padding_mode='circular')
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, padding_mode='circular')
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += residual
        x = self.relu(x)
        return x

class ResNet1D(nn.Module):
    def __init__(self, L=16, in_channels=1+1, out_channels=1, init_features=32, num_blocks=6):
        super().__init__()
        self.L = L
        features = init_features
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, features, kernel_size=5, padding=2, padding_mode='circular'),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.blocks = nn.Sequential(*[
            ResidualBlock1D(features, features)
            for _ in range(num_blocks)
        ])
        
        # Final output layer
        self.final = nn.Conv1d(features, out_channels, kernel_size=1)

    def forward(self, x, L=None):
        L = L if L is not None else self.L
        grid = get_grid(x, L)
        x = torch.cat((x.unsqueeze(-1), grid), dim=-1).permute(0, 2, 1)  # [B, 2, D]
        
        x = self.initial(x)
        x = self.blocks(x)
        x = self.final(x)
        
        return x.squeeze(1)  # [B, D]
    
class ResNet1D_RNN(nn.Module):
    def __init__(self, T_in=10, L=16,init_features=32,num_blocks=6):
        super().__init__()
        self.L = L
        self.initial = nn.Sequential(
            nn.Conv1d(T_in+1, init_features, kernel_size=5, padding=2, padding_mode='circular'),
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.Sequential(*[
            ResidualBlock1D(init_features, init_features)
            for _ in range(num_blocks)
        ])
        
        self.final = nn.Conv1d(init_features, 1, kernel_size=1)
    
    def forward(self, x, L=None):
        L = L if L is not None else self.L
        x = x.permute(0, 2, 1)  # [B, T, D] -> [B, D, T]
        grid = get_grid(x, L)
        x = torch.cat((x, grid), dim=-1).permute(0, 2, 1) #(B, T_in + 1, D)
        x = self.initial(x)
        x = self.blocks(x)
        x = self.final(x) # (B, 1, D) 
        return x 