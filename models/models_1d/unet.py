import torch
import torch.nn as nn
from .utils import get_grid

class UNet1D(nn.Module):
    def __init__(self, L=16, in_channels=1+1, out_channels=1, init_features=32):
        super(UNet1D, self).__init__()
        features = init_features
        self.L=L
        self.encoder1 = UNet1D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = UNet1D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.bottleneck = UNet1D._block(features * 2, features * 4, name="bottleneck")
        
        self.upconv2 = nn.ConvTranspose1d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet1D._block(features * 4, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose1d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet1D._block(features * 2, features, name="dec1")
        
        self.conv = nn.Conv1d(in_channels=features, out_channels=out_channels,kernel_size=1)

    def forward(self, x, L=None):
        L = L if L is not None else self.L
        grid = get_grid(x,L)
        x = torch.cat((x.unsqueeze(-1), grid), dim=-1).permute(0,2,1) # [B, 2, D]
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        
        bottleneck = self.bottleneck(self.pool2(enc2))
        
        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1) # [B, 1, D]
        
        return self.conv(dec1).squeeze(1) # [B, 1, D] -> [B, D]
 
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv1d(in_channels=in_channels,out_channels=features,kernel_size=5,padding=2,padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=features, out_channels=features, kernel_size=5,padding=2, padding_mode='circular'),
            nn.ReLU(inplace=True),
        )
    

class UNet1D_RNN(nn.Module):
    def __init__(self, T_in=10, L=16,init_features=32):
        super(UNet1D_RNN, self).__init__()
        self.T_in = T_in
        features = init_features
        self.L = L
        
        self.encoder1 = self._block(T_in + 1, features, name="enc1")  # +1 for grid
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.bottleneck = self._block(features * 2, features * 4, name="bottleneck")
        
        self.upconv2 = nn.ConvTranspose1d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(features * 4, features * 2, name="dec2")  # Skip connection doubles channels
        self.upconv1 = nn.ConvTranspose1d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")
        
        # Output layer
        self.conv = nn.Conv1d(features, 1, kernel_size=1)

    def forward(self, x, L=None):
        L = L if L is not None else self.L
        x = x.permute(0, 2, 1)  # [B, T, D] -> [B, D, T]
        grid = get_grid(x,L)
        x = torch.cat((x, grid), dim=-1).permute(0, 2, 1)   #(B, T_in + 1, D)
  
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        
        bottleneck = self.bottleneck(self.pool2(enc2))
        
        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)  # Skip connection
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)  # Skip connection
        dec1 = self.decoder1(dec1)
        
        # Output
        out = self.conv(dec1)  # (B, 1, D) 
        return out

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv1d(in_channels, features, kernel_size=5, padding=2, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv1d(features, features, kernel_size=5, padding=2, padding_mode='circular'),
            nn.ReLU(inplace=True),
        )