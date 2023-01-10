from torch import Tensor
import torch.nn as nn
import torch.nn.init as init

# 1. To scale down the features, pass through two Convolutional layers with 3x3 kernel size and stride of 2
class CNN_Layers(nn.Module):
    def __init__(self, out_channels=64, n_feats=128, kernel_size=3, stride=2):
        super(CNN_Layers, self).__init__()
        self.out_channels = out_channels
        self.n_feats = n_feats
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // stride

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.out_channels, kernel_size=self.kernel_size, stride = self.stride, padding = self.padding),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride = self.stride, padding = self.padding),
            nn.ReLU())
    
    def get_output_dim(self):
        output_dim = (self.n_feats + 2*self.padding - self.kernel_size) // self.stride + 1
        output_dim = (output_dim + 2*self.padding - self.kernel_size) // self.stride + 1
        return output_dim
            
    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.conv2(output)
        return output

class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialized by xavier initialization and bias initilize to Zeros
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)
        
    def forward(self, x: Tensor) -> Tensor:
        # Convert to the format to flatten
        output_sizes = x.size()
        output = x.view(output_sizes[0], output_sizes[1] * output_sizes[2], output_sizes[3])  # (batch, feature, seq_len)
        output = output.transpose(1, 2) # (batch, seq_len, feature)
        return self.linear(output)

class InputEncoder(nn.Module):
    """
    This InputEncoder encodes images before hand-over to the Transformer
    This Module consists of
        - 2 CNN layers (2 of Conv2d+ReLU)
        - M of Additional Modules (optional)
        - 1 Linear Layer to obtain embed_dim (Input Encoding)
        - embed_dim dimensional Positional Encoding is added to the Input Encoding
    """
    def __init__(self, n_feats: int = 128, embed_dim: int = 512) -> None:
        super(InputEncoder, self).__init__()
        self.cnn = CNN_Layers(out_channels=64, n_feats=n_feats, kernel_size=3, stride=2)
        self.linear = Linear(in_features = int(self.cnn.get_output_dim()*self.cnn.out_channels), 
                             out_features = embed_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        outputs = self.cnn(x)
        outputs = self.linear(outputs)
        return outputs       