import torch
import torch.nn as nn
from torch import Tensor
from models.modules.activations import Swish

class Transpose(nn.Module):
    def __init__(self, shape: tuple = (1, 2)):
        super().__init__()
        self.shape = shape
    
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.transpose(*self.shape)


class DepthWiseConv1d(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        kernel_size: int,
        bias: bool = False
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel_size must be an odd number to have the same output sequence length"
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(in_channels = in_channels, 
                                   out_channels = in_channels,
                                   kernel_size = kernel_size,
                                   stride = 1,
                                   padding = padding,
                                   bias = bias)
        
    def forward(self, intputs: Tensor) -> Tensor:
        return self.depthwise(intputs)
    
class PointWiseConv1d(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        bias: bool = True
    ) -> None:
        super().__init__()
        
        self.pointwise = nn.Conv1d(in_channels = in_channels,
                                   out_channels = out_channels,
                                   kernel_size = 1,
                                   stride = 1,
                                   padding = 0,
                                   bias = bias)
        
    def forward(self, intputs: Tensor) -> Tensor:
        return self.pointwise(intputs)

class Convolution(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        kernel_size: int,
        dropout_p: float = 0.1
    ) -> None:
        super().__init__()
        
        expansion_factor = 2
        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels), 
            Transpose(shape = (1, 2)),
            
            PointWiseConv1d(
                in_channels = in_channels, 
                out_channels = in_channels * expansion_factor,
                bias = True),
            
            nn.GLU(dim= 1),
            
            DepthWiseConv1d(
                in_channels = in_channels,
                kernel_size = kernel_size,
                bias = False),
            
            nn.BatchNorm1d(in_channels),
            
            Swish(),
            
            PointWiseConv1d(
                in_channels = in_channels, 
                out_channels = in_channels,
                bias = True
            ),
            
            nn.Dropout(p=dropout_p)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs).transpose(1, 2)