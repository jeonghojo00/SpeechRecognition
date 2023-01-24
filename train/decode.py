import torch
from typing import List

def GreedyDecoder(text_transform, output, labels, label_lengths, blank_label=0, collapse_repeated=True):
    """Decodes a batch of output into a batch of texts"""
    #output :  (batch, time, n_class)
    #labels :  (batch, time)
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text_sequence(
				labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text_sequence(decode))
        return decodes, targets

def Decoder(text_transform, output, blank_label=0, collapse_repeated=True):
    """Decodes single output into a text for inference"""    
    arg_maxes = torch.argmax(output, dim=2)
    for i, args in enumerate(arg_maxes):
        decode = []
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decoded = text_transform.int_to_text(decode)
    return decoded


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels   # List of classes
        self.blank = blank     # blank label index

    def forward(self, emission: torch.Tensor, predicted=True) -> List[str]:
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        if predicted:
            indices = torch.argmax(emission, dim=-1)  # [num_seq,]
            indices = torch.unique_consecutive(indices, dim=-1) # Delete Repeated characters
        else:
            indices = emission
        indices = [i for i in indices if i != self.blank]   # Remove blank label
        joined = "".join([self.labels[i] for i in indices]) # Convert the indices into the labels
        return joined.replace("|", " ").strip().split()