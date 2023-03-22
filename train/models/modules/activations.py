import torch
import torch.nn as nn

class Swish(nn.Module):
    """
    Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks applied to a variety of challenging domains such as Image classification and Machine translation (https://medium.com/@neuralnets/swish-activation-function-by-google-53e1ea86f820)
    As beta closes to 1, Swish is equivalent to the Sigmoid-weighted Linear Unit (Sil)
    As beta closes to 0, Swish becomes the sacaled linear function
    """
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta
    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)