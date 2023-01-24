import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.init as init
from typing import Tuple

# 1. To scale down the features, pass through two Convolutional layers with 3x3 kernel size and stride of 2
class CNN_Layers(nn.Module):
    def __init__(self, out_channels=256, n_feats=80):
        super(CNN_Layers, self).__init__()
        self.out_channels = out_channels
        self.n_feats = n_feats
        self.kernel_size = 3
        self.stride = 2
        self.padding = 3 // 2 # kernel // stride

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


class VGGnet(nn.Module):
    """
    # Reference: Advances in Joint CTC-Attention based End-to-End Speech Recognition with a Deep CNN Encoder and RNN-LM (JUN 2017)
    # - Normalization is not need as it does not imporove the performance in deep CNN according to the paper.
    # - Scale down to 1/4 through the VGGnet
    # Reference2: Very Deep Convolutional Networks for Large-Scale Image Recognition
    # Reduce the input spectrogram into 1/4 size
    """
    def __init__(self, out_channels=128, n_feats=80):
        super(VGGnet, self).__init__()
        self.out_channels = out_channels
        self.n_feats = n_feats
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.conv_4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
    
    def get_output_dim(self, input_dim):
        # Maxpooling decreases the input dimension while dimensions through cnn stay the same.
        output_dim = ((input_dim + (2*self.maxpool.padding) - self.maxpool.kernel_size) // self.maxpool.stride) + 1
        output_dim = ((output_dim + (2*self.maxpool.padding) - self.maxpool.kernel_size) // self.maxpool.stride) + 1
        return output_dim
        
    def forward(self, x):
        """
        input       x: (batch, channel, n_feature, seq_len)
        output output: (batch, channel, 1/4 * n_feature, 1/4 * seq_len)
        """
        output = self.conv_1(x)
        output = self.conv_2(output)
        output = self.maxpool(output)
        output = self.conv_3(output)
        output = self.conv_4(output)
        output = self.maxpool(output)  
        return output

class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialized by xavier initialization and bias initilize to Zeros
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        
        if bias:
            init.zeros_(self.linear.bias)
        
    def forward(self, x: Tensor) -> Tensor:
        # Convert to the format to flatten
        output_sizes = x.size()                                                               # (batch, n_channels, n_feature, seq_len)
        output = x.view(output_sizes[0], output_sizes[1] * output_sizes[2], output_sizes[3])  # (batch, feature, seq_len)
        output = output.transpose(1, 2) # (batch, seq_len, feature)
        return self.linear(output)

class InputEncoder(nn.Module):
    """
    This InputEncoder encodes images before hand-over to the Transformer (convolutional subsampling)
    This Module consists of
        - Convolution Subsampling (2 cnn layers or VGG)
        - M of Additional Modules (optional)
        - 1 Linear Layer to obtain embed_dim (Input Encoding)
        - embed_dim dimensional Positional Encoding is added to the Input Encoding
    """
    def __init__(self, conv_subsampling: str='vgg', n_feats: int = 80, embed_dim: int = 256, dropout_p: float = 0.1) -> None:
        super(InputEncoder, self).__init__()
        self.conv_subsampling = conv_subsampling
        if conv_subsampling == 'vgg':
            self.cnn = VGGnet(out_channels=128, n_feats=n_feats)
        else:
            self.cnn = CNN_Layers(out_channels=256, n_feats=n_feats)
        self.linear = Linear(in_features = int(self.cnn.get_output_dim(n_feats)*self.cnn.out_channels), 
                             out_features = embed_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        
    def forward(self, x: Tensor, x_lens: Tensor) -> Tuple[Tensor, Tensor]:
        output = self.cnn(x)
        output = self.linear(output)
        output_lens = x_lens
        if self.conv_subsampling == "vgg":
            for i in range(len(output_lens)):
                output_lens[i] = self.cnn.get_output_dim(output_lens[i])
        else:
            output_lens = x_lens >> 2
            output_lens -= 1
        
        return self.dropout(output), output_lens   