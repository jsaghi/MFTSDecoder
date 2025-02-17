import torch
import torch.nn as nn

# Class DenseFilterExpansion
class DenseFilterExpansion(nn.Module):
  def __init__(self, output_shape):
    super().__init__()
    self.output_seq_length = output_shape[-1]
    self.num_filters = output_shape[-2]
    self.w = nn.Parameter(torch.randn(self.output_seq_length, self.num_filters))
    self.b = nn.Parameter(torch.zeros(self.output_seq_length, self.num_filters))

  def forward(self, inputs):
    in_slice = torch.split(inputs, 1, dim=-1)
    w_slice = torch.split(self.w, 1, dim=0)
    slices = []
    for i in range(self.output_seq_length):
      slices.append(torch.matmul(in_slice[i], w_slice[i]))
    outputs = torch.stack(slices, axis=1)
    outputs = torch.squeeze(outputs, dim=-2)
    outputs = outputs + self.b
    outputs = torch.transpose(outputs, -1, -2)
    return outputs
  

# Decoders
def decoder(input_shape):
  decoder = nn.Sequential(
    DenseFilterExpansion((2048, input_shape[-1])),
    nn.Tanh(),
    nn.ConvTranspose1d(in_channels=2048, out_channels=1024,
                       kernel_size=3, stride=1, padding=0),
    nn.ConvTranspose1d(in_channels=1024, out_channels=512,
                       kernel_size=5, stride=2, padding=2, output_padding=1),
    nn.ConvTranspose1d(in_channels=512, out_channels=256,
                       kernel_size=5, stride=2, padding=2, output_padding=1),
    nn.ConvTranspose1d(in_channels=256, out_channels=128,
                       kernel_size=3, stride=1, padding=0),
    nn.ConvTranspose1d(in_channels=128, out_channels=64,
                       kernel_size=5, stride=2, padding=2, output_padding=1),
    nn.ConvTranspose1d(in_channels=64, out_channels=32,
                       kernel_size=5, stride=2, padding=2, output_padding=1),
    nn.ConvTranspose1d(in_channels=32, out_channels=1,
                       kernel_size=5, stride=2, padding=2, output_padding=1),
    )
  return decoder
