"""
1D models for neural network architectures
"""
from .fno import FNO1D, FNO1D_RNN
from .unet import UNet1D, UNet1D_RNN
from .tinyCNN import TinyCNNNet
from .deeponet import DeepONet1D, DeepONet1D_RNN, DeepONet1D_GRU
from .resnet import ResNet1D, ResNet1D_RNN
from .unetmod import UNetMod1D, UNetMod1D_RNN
from .utils import *

__all__ = [
    'FNO1D', 'FNO1D_RNN',
    'UNet1D', 'UNet1D_RNN', 
    'TinyCNNNet',
    'DeepONet1D', 'DeepONet1D_RNN', 'DeepONet1D_GRU',
    'ResNet1D', 'ResNet1D_RNN',
    'UNetMod1D', 'UNetMod1D_RNN'
]