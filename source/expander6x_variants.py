from custom_layers import DenseFilterExpansion, FilterAttention, ConvInsert
import torch
import torch.nn as nn
import torcn.nn.functional as F


# Decoder class to expand by a ratio of 6 to 1
class Expander6XBase(nn.Module):
  def __init__(self, input_shape):
    super().__init__()
    self.conv1 = nn.ConvTranspose1d(in_channels=1, out_channels=64, kernel_size=5,
                      stride=2, padding=2, output_padding=1)
    self.fa1 = FilterAttention(64, input_shape[-1] * 2)
    self.ci1 = ConvInsert(336, 8)
    self.conv2 = nn.ConvTranspose1d(in_channels=1, out_channels=64,
                       kernel_size=3, stride=1, padding=0)
    self.conv3 = nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=5,
                      stride=2, padding=2, output_padding=1)
    self.fa2 = FilterAttention(128, 1008)

  def forward(self, inputs):
    outputs = self.conv1(inputs)
    outputs = self.fa1(outputs)
    outputs = self.ci1(outputs)
    outputs = self.conv2(outputs)
    outputs = self.conv3(outputs)
    outputs = self.fa2(outputs)
    return outputs
  

# Model which adds the dense filter expansion layer before the first convolutonal layer
class Expander6XDFE(nn.Module):
  def __init__(self, input_shape):
    super().__init__()
    self.dfe1 = DenseFilterExpansion(num_filters=32, seq_length=input_shape[-1])
    self.conv1 = nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=5,
                      stride=2, padding=2, output_padding=1)
    self.fa1 = FilterAttention(64, input_shape[-1] * 2)
    self.ci1 = ConvInsert(336, 8)
    self.conv2 = nn.ConvTranspose1d(in_channels=1, out_channels=64,
                       kernel_size=3, stride=1, padding=0)
    self.conv3 = nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=5,
                      stride=2, padding=2, output_padding=1)
    self.fa2 = FilterAttention(128, 1008)

  def forward(self, inputs):
    outputs = self.dfe1(inputs)
    outputs = self.conv1(outputs)
    outputs = self.fa1(outputs)
    outputs = self.ci1(outputs)
    outputs = self.conv2(outputs)
    outputs = self.conv3(outputs)
    outputs = self.fa2(outputs)
    return outputs