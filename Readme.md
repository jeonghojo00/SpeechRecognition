# Speech Recognition

This repository contains implementations of the Deep Speech 2, Transformer, and Conformer models for speech recognition.

## Deep Speech 2

- The Deep Speech 2 model is a deep recurrent neural network designed to transcribe speech input into text output. It is based on the original Deep Speech model developed by Baidu Research, but with several improvements that make it more accurate and efficient.
- This model increased the model capacity via depth because simple multi-layer with a single RNN cannot exploit thousands of hours of labelled speech data.
- This model is basically CNN-LSTM model, composed of 1 to 3 CNN layers, 1 to 7 Bidirectional recurrent or GRU layers, and one or more Fully-Connected layers at the end.
- This model is trained using CTC loss function.
<img src="https://github.com/jeonghojo00/SpeechRecognition/blob/main/images/DeepSpeech2_Architecture.png" width="300">

## Transformer

- The Transformer model is a neural network architecture that was introduced in the paper "Attention is All You Need" by Vaswani et al. It has been shown to be effective for a variety of natural language processing tasks, including speech recognition.
- Transformer requires bit more iterations than deep speech 2 until convergence but shows better performance in WER.
- Current SOTA Transformer model uses Joint CTC-CE Loss.
<img src="https://github.com/jeonghojo00/SpeechRecognition/blob/main/images/Transformer_Architecture.png" width="300">


## Conformer

- The Conformer model is an extension of the Transformer model that includes convolutional layers in addition to the self-attention mechanism. This is intended to improve the model's ability to handle variations in the duration of the input speech.
- It implements relative positioning encoding inside multi-head attention, which is different from transformer model that implements fixed positioning encoding before the attention layer.
- This model is trained using CTC loss function.
<img src="https://github.com/jeonghojo00/SpeechRecognition/blob/main/images/Conformer_Architecture.png" width="300">

## Usage

To train the models, you will need to have a Librispeech dataset of speech and corresponding transcriptions. This dataset and required parameters can be loaded utilizing ini files. Once you load the ini files, you can use the provided scripts to train the models. The scripts include options for specifying the dataset, model architecture, and training parameters.

## References
### Deep Speech 2
Amodei, Dario, et al. "Deep speech 2: End-to-end speech recognition in English and Mandarin." arXiv preprint arXiv:1512.02595 (2015).
### Transformer
Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems. 2017.
### Conformer
Gao, Zhiqiang, et al. "Conformer: Convolution-augmented transformer for speech recognition." arXiv preprint arXiv:2005.08100 (2020).
Dai, Zihang, et al. "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" arXiv preprint arXiv:1901.02860 (2019).
