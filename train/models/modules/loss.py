import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LabelSmoothedCrossEntropyLoss(nn.Module):
    """
    Label smoothed cross entropy loss function.
    Args:
        num_classes (int): the number of classfication
        ignore_index (int): Indexes that are ignored when calculating loss
        smoothing (float): ratio of smoothing (confidence = 1.0 - smoothing)
        dim (int): dimension of calculation loss
        reduction (str): reduction method [sum, mean] (default: sum)
        architecture (str): speech model`s model [las, transformer] (default: las)
    Inputs: logits, target
        logits (torch.Tensor): probability distribution value from model and it has a logarithm shape
        target (torch.Tensor): ground-thruth encoded to integers which directly point a word in label
    Returns: label_smoothed
        - **label_smoothed** (float): sum of loss
    """
    def __init__(
            self,
            num_classes: int,           # the number of classfication
            ignore_index: int,          # indexes that are ignored when calcuating loss
            smoothing: float = 0.1,     # ratio of smoothing (confidence = 1.0 - smoothing)
            dim: int = -1,              # dimension of caculation loss
            reduction='sum',            # reduction method [sum, mean]
            architecture='las',         # speech model`s model [las, transformer]
    ) -> None:
        super(LabelSmoothedCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.dim = dim
        self.ignore_index = ignore_index
        self.reduction = reduction.lower()
        self.architecture = architecture.lower()

        if self.reduction == 'sum':
            self.reduction_method = torch.sum
        elif self.reduction == 'mean':
            self.reduction_method = torch.mean
        else:
            raise ValueError("Unsupported reduction method {0}".format(reduction))

    def forward(self, logits: Tensor, targets: Tensor):
        if self.architecture == 'transformer':
            logits = F.log_softmax(logits, dim=-1)

        if self.smoothing > 0.0:
            with torch.no_grad():
                label_smoothed = torch.zeros_like(logits)
                label_smoothed.fill_(self.smoothing / (self.num_classes - 1))
                label_smoothed.scatter_(1, targets.data.unsqueeze(1), self.confidence)
                label_smoothed[targets == self.ignore_index, :] = 0
            return self.reduction_method(-label_smoothed * logits)

        return F.cross_entropy(logits, targets, ignore_index=self.ignore_index, reduction=self.reduction)
    
from torch import Tensor
from typing import Tuple
import torch.nn as nn
IGNORE_ID = 0  # indexes that are ignored when calcuating loss

# Joint CTC Cross Entropy
class Joint_CTC_CrossEntropy_Loss(nn.Module):
    def __init__(self,
                 num_classes: int,                                # the number of classfication                             
                 dim: int = -1,                                   # dimension of caculation loss
                 reduction: str = 'mean',                         # reduction method ['sum', 'mean', 'elementwise_mean']
                 smoothing: float = 0.1,                          # ratio of smoothing (confidence = 1.0 - smoothing)
                 ctc_weight: float = 0.5,                         # ratio of ctc loss (cross entropy loss ratio = 1.0 - ctc loss ratio)
                 blank_id: int = 0,
                 architecture: str = 'transformer'
                 ) -> None:
        super(Joint_CTC_CrossEntropy_Loss, self).__init__()
        
        self.ctc_weight = ctc_weight
        self.ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)
        if smoothing > 0.0:
            self.ce_loss = LabelSmoothedCrossEntropyLoss(
                                num_classes = num_classes,        # the number of classfication
                                ignore_index = IGNORE_ID,         # indexes that are ignored when calcuating loss
                                smoothing = smoothing,            # ratio of smoothing (confidence = 1.0 - smoothing)
                                dim = dim,                        # dimension of caculation loss
                                reduction = reduction,            # reduction method ['sum', 'mean', 'elementwise_mean']
                                architecture = architecture)
        else:
            self.ce_loss = nn.CrossEntropyLoss(
                                ignore_index = IGNORE_ID,
                                reduction=reduction)
    
    def forward(self,
                encoder_log_probs: Tensor,
                encoder_output_lens: Tensor,
                output_softmax: Tensor,
                target: Tensor,
                target_lens: Tensor
                ) -> Tuple[Tensor, Tensor, Tensor]:
        
        decoder_log_probs = output_softmax.contiguous().view(-1, output_softmax.size(-1)) # (batch * time, n_class)

        ctc_loss = self.ctc_loss(encoder_log_probs, target, encoder_output_lens, target_lens)
        ce_loss = self.ce_loss(decoder_log_probs, target.contiguous().view(-1))

        loss = self.ctc_weight * ctc_loss + (1.0 - self.ctc_weight) * ce_loss
        return loss, ctc_loss, ce_loss