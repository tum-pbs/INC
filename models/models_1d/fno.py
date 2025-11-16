import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_grid

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1D(nn.Module):
    def __init__(self, L=16, modes=16, width=64):
        super(FNO1D, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (B,D) -> (B, D,2) by adding location information
        output: the solution of a later timestep
        output shape: (B,D,1).sqeeze(-1) -> (B,D)
        """

        self.modes1 = modes
        self.width = width
        self.L = L
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, L=None):
        L = L if L is not None else self.L
        grid = get_grid(x,L)
        x = torch.cat((x.unsqueeze(-1), grid), dim=-1) # [B, D, 2]
        x = self.fc0(x)
        x = x.permute(0, 2, 1) # [B, width, D]
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1) # [B, D, width]
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x) # [B, D, 1]
        return x.squeeze(-1)


class FNO1D_RNN(nn.Module):
    def __init__(self, modes=16, width=64, T_in=10, L=16):
        """
        FNO1d with a recurrent (autoregressive) structure.
        
        Args:
            modes: number of Fourier modes to use.
            width: channel width after the lifting layer.
            T_in: number of initial time steps.
                   The network will take an input of shape (B, D, T_in)
                   and predict one future time step at a time.
        """
        super(FNO1D_RNN, self).__init__()
        self.modes1 = modes
        self.width = width
        self.T_in = T_in
        self.L = L
        # self.padding = 2  # if input is non-periodic
        # fc0 now lifts (T_in + 1) channels (T_in past time steps + 1 grid channel)
        self.fc0 = nn.Linear(T_in + 1, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, L=None):
        """
        Autoregressively predict T_future time steps.
        
        Args:
            x: initial input tensor of shape (B, D, T_in), where D is the spatial dimension.
            T_future: number of time steps to predict.
            step: prediction step size (usually 1).
        
        Returns:
            A tensor of shape (B, 1, D) containing the future predictions.
        """
        L = L if L is not None else self.L
        x = x.permute(0, 2, 1)  # [B, T, D] -> [B, D, T]
        grid = get_grid(x,L)
        x = torch.cat((x, grid), dim=-1) # [B, D, T_in+1], 1 extra channel for grid
        x = self.fc0(x) # [B, D, width]
        x = x.permute(0, 2, 1)  # [B, D, width] -> [B, width, D]
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = x.permute(0, 2, 1) # [B, D, width]
        x = self.fc1(x) 
        x = F.gelu(x)
        x = self.fc2(x) # [B, D, 1]
        return x.permute(0, 2, 1) # [B, 1, D]