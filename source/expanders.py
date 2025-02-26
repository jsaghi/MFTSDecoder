from settings import *
from custom_layers import DenseFilterExpansion, FilterAttention
import torch
import torch.nn as nn
import torch.nn.functional as F

# Class to expand an input by a ratio of 36 to 1
class Expander36X(nn.Module):
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
    self.conv9 = nn.ConvTranspose1d(in_channels=16, out_channels=1,
                       kernel_size=5, stride=2, padding=2, output_padding=1)

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
    return outputs
  

# Decoder class to expand by a ratio of 6 to 1
class Expander6X(nn.Module):
  def __init__(self, input_shape):
    super().__init__()
    self.dfe = DenseFilterExpansion(4096, input_shape[-1])
    self.conv1 = nn.ConvTranspose1d(in_channels=4096, out_channels=2048,
                       kernel_size=11, stride=1, padding=0)
    self.conv2 = nn.ConvTranspose1d(in_channels=2048, out_channels=1024,
                       kernel_size=11, stride=1, padding=0)
    self.conv3 = nn.ConvTranspose1d(in_channels=1024, out_channels=512,
                       kernel_size=11, stride=1, padding=0)
    self.conv4 = nn.ConvTranspose1d(in_channels=512, out_channels=256,
                       kernel_size=11, stride=1, padding=0)
    self.conv5 = nn.ConvTranspose1d(in_channels=256, out_channels=128,
                       kernel_size=11, stride=1, padding=0)
    self.conv6 = nn.ConvTranspose1d(in_channels=128, out_channels=64,
                       kernel_size=11, stride=1, padding=0)
    self.conv7 = nn.ConvTranspose1d(in_channels=64, out_channels=32,
                       kernel_size=11, stride=1, padding=0)
    self.conv8 = nn.ConvTranspose1d(in_channels=32, out_channels=16,
                       kernel_size=11, stride=1, padding=0)
    self.conv9 = nn.ConvTranspose1d(in_channels=16, out_channels=8,
                       kernel_size=5, stride=1, padding=0)
    self.conv10 = nn.ConvTranspose1d(in_channels=8, out_channels=4,
                       kernel_size=5, stride=2, padding=2, output_padding=1)
    self.conv11 = nn.ConvTranspose1d(in_channels=4, out_channels=1,
                       kernel_size=5, stride=2, padding=2, output_padding=1)

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
    outputs = self.conv10(outputs)
    outputs = self.conv11(outputs)
    return outputs
  

# Class that stacks 5 Expader36X modules, splits inputs along features, feeds
# each feature through a separate expander, and then stacks the outputs again
class LFExpanderStack(nn.Module):
  def __init__(self, num_features):
    super().__init__()
    self.expander_stack = [Expander36X((1, LF_LENGTH))] * num_features

  def forward(self, inputs):
    in_slices = torch.split(inputs, 1, dim=-1)
    out_slices = []
    for i, slice in enumerate(in_slices):
      out_slices.append(self.expander_stack[i](slice.transpose(-1, -2)))
    return torch.squeeze(torch.stack(out_slices, dim=-1), dim=1)
  

# Class that stacks 4 Expader6X modules, splits inputs along features, feeds
# each feature through a separate expander, and then stacks the outputs again
class IFExpanderStack(nn.Module):
  def __init__(self, num_features):
    super().__init__()
    self.expander_stack = [Expander6X((1, IF_LENGTH))] * num_features

  def forward(self, inputs):
    in_slices = torch.split(inputs, 1, dim=-1)
    in_slices = torch.transpose()
    out_slices = []
    for i, slice in enumerate(in_slices):
      out_slices.append(self.expander_stack[i](slice.transpose(-1, -2)))
    return torch.squeeze(torch.stack(out_slices, dim=-1), dim=1)
  