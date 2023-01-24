import torch
import torch.nn as nn
import torchaudio
from typing import List

LABELS        = "|abcdefghijklmnopqrstuvwxyz'"

# Text Transform
class TextTransform:
    """ Maps Characters -> Integers and vice versa"""
    def __init__(self, classes):
        classes = list(classes)
        self.char2index = {}
        self.index2char = {}
        for i, char in enumerate(classes):
            self.index2char[i] = char
            self.char2index[char] = i

    def text_to_int(self, text):
        """ Convert text into an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char2index['|']
            else:
                ch = self.char2index[c]
            int_sequence.append(ch)
        return int_sequence
    
    def int_to_text(self, labels):
        """ Convert integer labels to a text sequenc """
        if type(labels) == torch.Tensor:
            labels = labels.tolist()
        text = []
        for i in labels:
            text.append(self.index2char[i])
        return ''.join(text).replace('|', ' ')

text_transform = TextTransform(LABELS)

# Collate Function for Data Transform for DataLoader
def collate_fn(batch, train=True):
    '''
    input: data: a batch of dataset
    output: 
        - spectrograms: (batch, 1, n_mels, framerate)
        - labels:  (batch, longest label length in the batch)
        - input_lengths: (batch), each number represents each framerate before padding
        - label_lenghts: (batch), each number represents each label length before padding
    '''
    
    SAMPLE_RATE   = 16000    # 16,000 samples per second
    N_FILTERBANKS = 80     # Number of filterbanks (= number of features to represent)
    WINDOW_SIZE   = 25e-3    # 25ms duration of a window
    STRIDE_SIZE   = 10e-3    # 10ms duration of a stride
    N_FFT         = int(SAMPLE_RATE * WINDOW_SIZE)    # number of samples per window
    HOP_LENS      = int(SAMPLE_RATE * STRIDE_SIZE) # number of samples per stride
    mel = torchaudio.transforms.MelSpectrogram(sample_rate = SAMPLE_RATE, 
                                               n_mels      = N_FILTERBANKS,
                                               n_fft       = N_FFT, 
                                               hop_length  = HOP_LENS)
    frequency_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=27)
    p_s = 0.05     # maximum time-mask ratio
    
    specs      = []
    labels     = []
    spec_lens  = []
    label_lens = []
    for (waveform, _, utterance, _, _, _) in batch:
        spec = mel(waveform)
        if train:
            spec = frequency_mask(spec)
            for _ in range(10):  
                spec = torchaudio.transforms.TimeMasking(time_mask_param=int(p_s * len(utterance)),
                                                         p = p_s)(spec)
        specs.append(spec.squeeze(0).transpose(0, 1))                                       # (channel, n_mels, seq_len) -> (seq_len, n_mels)
        
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        
        spec_lens.append(spec.shape[2])
        label_lens.append(len(label))
    
    specs = nn.utils.rnn.pad_sequence(specs, batch_first=True).unsqueeze(1).transpose(2, 3) # -> (channel, n_mels, max_seq_len of batch)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0.).long()
    
    return specs, labels, torch.tensor(spec_lens), torch.tensor(label_lens)
