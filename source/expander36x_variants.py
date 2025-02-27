from settings import *
from custom_layers import DenseFilterExpansion, FilterAttention
import torch
import torch.nn as nn
import torch.nn.functional as F


# Variant that consists of only conv layers
class Expander36XFA(nn.Module):
  def __init__(self, input_shape):
    super().__init__()
    self.conv1 = nn.ConvTranspose1d(in_channels=1, out_channels=4096,
                       kernel_size=3, stride=1, padding=0)
    self.conv2 = nn.ConvTranspose1d(in_channels=4096, out_channels=2048,
                       kernel_size=3, stride=1, padding=0)
    self.conv3 = nn.ConvTranspose1d(in_channels=2048, out_channels=1024,
                       kernel_size=5, stride=2, padding=2, output_padding=1)
    self.conv4 = nn.ConvTranspose1d(in_channels=1024, out_channels=512,
                       kernel_size=5, stride=2, padding=2, output_padding=1)
    self.conv5 = nn.ConvTranspose1d(in_channels=512, out_channels=256,
                       kernel_size=3, stride=1, padding=0)
    self.conv6 = nn.ConvTranspose1d(in_channels=256, out_channels=128,
                       kernel_size=3, stride=1, padding=0)
    self.conv7 = nn.ConvTranspose1d(in_channels=128, out_channels=64,
                       kernel_size=3, stride=1, padding=0)
    self.conv8 = nn.ConvTranspose1d(in_channels=64, out_channels=32,
                       kernel_size=5, stride=2, padding=2, output_padding=1)
    self.conv9 = nn.ConvTranspose1d(in_channels=32, out_channels=16,
                       kernel_size=5, stride=2, padding=2, output_padding=1)
    self.conv10 = nn.ConvTranspose1d(in_channels=16, out_channels=8,
                       kernel_size=5, stride=2, padding=2, output_padding=1)

  def forward(self, inputs):
    outputs = self.conv1(inputs)
    outputs = self.conv2(outputs)
    outputs = self.conv3(outputs)
    outputs = self.conv4(outputs)
    outputs = self.conv5(outputs)
    outputs = self.conv6(outputs)
    outputs = self.conv7(outputs)
    outputs = self.conv8(outputs)
    outputs = self.conv9(outputs)
    outputs = F.sigmoid(self.conv10(outputs))
    return outputs


# Variant that drops the dense filter expansion layer but keeps the filter attention layer
class Expander36XFA(nn.Module):
  def __init__(self, input_shape):
    super().__init__()
    self.conv1 = nn.ConvTranspose1d(in_channels=1, out_channels=4096,
                       kernel_size=3, stride=1, padding=0)
    self.conv2 = nn.ConvTranspose1d(in_channels=4096, out_channels=2048,
                       kernel_size=3, stride=1, padding=0)
    self.conv3 = nn.ConvTranspose1d(in_channels=2048, out_channels=1024,
                       kernel_size=5, stride=2, padding=2, output_padding=1)
    self.conv4 = nn.ConvTranspose1d(in_channels=1024, out_channels=512,
                       kernel_size=5, stride=2, padding=2, output_padding=1)
    self.conv5 = nn.ConvTranspose1d(in_channels=512, out_channels=256,
                       kernel_size=3, stride=1, padding=0)
    self.conv6 = nn.ConvTranspose1d(in_channels=256, out_channels=128,
                       kernel_size=3, stride=1, padding=0)
    self.conv7 = nn.ConvTranspose1d(in_channels=128, out_channels=64,
                       kernel_size=3, stride=1, padding=0)
    self.conv8 = nn.ConvTranspose1d(in_channels=64, out_channels=32,
                       kernel_size=5, stride=2, padding=2, output_padding=1)
    self.conv9 = nn.ConvTranspose1d(in_channels=32, out_channels=16,
                       kernel_size=5, stride=2, padding=2, output_padding=1)
    self.conv10 = nn.ConvTranspose1d(in_channels=16, out_channels=8,
                       kernel_size=5, stride=2, padding=2, output_padding=1)
    self.fa = FilterAttention(8, SEQ_LENGTH)

  def forward(self, inputs):
    outputs = self.conv1(inputs)
    outputs = self.conv2(outputs)
    outputs = self.conv3(outputs)
    outputs = self.conv4(outputs)
    outputs = self.conv5(outputs)
    outputs = self.conv6(outputs)
    outputs = self.conv7(outputs)
    outputs = self.conv8(outputs)
    outputs = self.conv9(outputs)
    outputs = self.conv10(outputs)
    outputs = F.sigmoid(self.fa(outputs))
    return outputs


# Variant that integrates the filter attention layer and the dfe layer
class Expander36XFA(nn.Module):
  def __init__(self, input_shape):
    super().__init__()
    self.dfe = DenseFilterExpansion(4096, input_shape[-1])
    self.conv1 = nn.ConvTranspose1d(in_channels=4096, out_channels=2048,
                       kernel_size=3, stride=1, padding=0)
    self.conv2 = nn.ConvTranspose1d(in_channels=2048, out_channels=1024,
                       kernel_size=5, stride=2, padding=2, output_padding=1)
    self.conv3 = nn.ConvTranspose1d(in_channels=1024, out_channels=512,
                       kernel_size=5, stride=2, padding=2, output_padding=1)
    self.conv4 = nn.ConvTranspose1d(in_channels=512, out_channels=256,
                       kernel_size=3, stride=1, padding=0)
    self.conv5 = nn.ConvTranspose1d(in_channels=256, out_channels=128,
                       kernel_size=3, stride=1, padding=0)
    self.conv6 = nn.ConvTranspose1d(in_channels=128, out_channels=64,
                       kernel_size=3, stride=1, padding=0)
    self.conv7 = nn.ConvTranspose1d(in_channels=64, out_channels=32,
                       kernel_size=5, stride=2, padding=2, output_padding=1)
    self.conv8 = nn.ConvTranspose1d(in_channels=32, out_channels=16,
                       kernel_size=5, stride=2, padding=2, output_padding=1)
    self.conv9 = nn.ConvTranspose1d(in_channels=16, out_channels=8,
                       kernel_size=5, stride=2, padding=2, output_padding=1)
    self.fa = FilterAttention(8, SEQ_LENGTH)

  def forward(self, inputs):
    outputs = F.tanh(self.dfe(inputs))
    outputs = self.conv1(outputs)
    outputs = self.conv2(outputs)
    outputs = self.conv3(outputs)
    outputs = self.conv4(outputs)
    outputs = self.conv5(outputs)
    outputs = self.conv6(outputs)
    outputs = self.conv7(outputs)
    outputs = self.conv8(outputs)
    outputs = self.conv9(outputs)
    outputs = F.sigmoid(self.fa(outputs))
    return outputs
  