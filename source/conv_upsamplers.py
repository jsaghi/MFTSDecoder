from settings import *
from custom_layers import DenseFilterExpansion, FilterAttention, ConvInsert
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


# Class to expand an input by a ratio of 36 to 1
class Expander36X(nn.Module):
  def __init__(self, out_length):
    super().__init__()
    self.conv1 = nn.ConvTranspose1d(in_channels=1, out_channels=1024,
                       kernel_size=5, stride=2, padding=2, output_padding=1)
    self.conv2 = nn.ConvTranspose1d(in_channels=1024, out_channels=512,
                       kernel_size=5, stride=2, padding=2, output_padding=1)
    self.fa1 = FilterAttention(512, out_length // 9)
    self.ci1 = ConvInsert(out_length // 9, 32)
    self.conv3 = nn.ConvTranspose1d(in_channels=1, out_channels=8,
                       kernel_size=3, stride=1, padding=0)
    self.conv4 = nn.ConvTranspose1d(in_channels=8, out_channels=2048,
                       kernel_size=5, stride=2, padding=2, output_padding=1)
    self.conv5 = nn.ConvTranspose1d(in_channels=2048, out_channels=1024,
                       kernel_size=5, stride=2, padding=2, output_padding=1)
    self.conv6 = nn.ConvTranspose1d(in_channels=1024, out_channels=512,
                       kernel_size=5, stride=2, padding=2, output_padding=1)
    self.fa2 = FilterAttention(512, out_length)

  def forward(self, inputs):
    outputs = self.conv1(inputs)
    outputs = self.conv2(outputs)
    outputs = self.fa1(outputs)
    outputs = self.ci1(outputs)
    outputs = self.conv3(outputs)
    outputs = self.conv4(outputs)
    outputs = self.conv5(outputs)
    outputs = self.conv6(outputs)
    outputs = self.fa2(outputs)
    outputs = F.sigmoid(outputs)
    return outputs
  

# Decoder class to expand by a ratio of 6 to 1
class Expander6X(nn.Module):
  def __init__(self, out_length):
    super().__init__()
    self.conv1 = nn.ConvTranspose1d(in_channels=1, out_channels=64, kernel_size=5,
                      stride=2, padding=2, output_padding=1)
    self.fa1 = FilterAttention(64, out_length // 3)
    self.ci1 = ConvInsert(out_length // 3, 8)
    self.conv2 = nn.ConvTranspose1d(in_channels=1, out_channels=64,
                       kernel_size=3, stride=1, padding=0)
    self.conv3 = nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=5,
                      stride=2, padding=2, output_padding=1)
    self.fa2 = FilterAttention(128, out_length)

  def forward(self, inputs):
    outputs = self.conv1(inputs)
    outputs = self.fa1(outputs)
    outputs = self.ci1(outputs)
    outputs = self.conv2(outputs)
    outputs = self.conv3(outputs)
    outputs = self.fa2(outputs)
    outputs = F.sigmoid(outputs)
    return outputs
  

# Class that stacks 5 Expader36X modules, splits inputs along features, feeds
# each feature through a separate expander, and then stacks the outputs again
class LFExpanderStack(nn.Module):
  def __init__(self, num_features, out_length):
    super().__init__()
    self.expander_stack = nn.ModuleList([Expander36X(out_length) for _ in range(num_features)])

  def forward(self, inputs):
    in_slices = torch.split(inputs, 1, dim=-1)
    out_slices = []
    for i, slice in enumerate(in_slices):
      out_slices.append(self.expander_stack[i](slice.transpose(-1, -2)))
    return torch.squeeze(torch.stack(out_slices, dim=-1), dim=1)
  

# Class that stacks 4 Expader6X modules, splits inputs along features, feeds
# each feature through a separate expander, and then stacks the outputs again
class IFExpanderStack(nn.Module):
  def __init__(self, num_features, out_length):
    super().__init__()
    self.expander_stack = nn.ModuleList([Expander6X(out_length) for _ in range(num_features)])

  def forward(self, inputs):
    in_slices = torch.split(inputs, 1, dim=-1)
    out_slices = []
    for i, slice in enumerate(in_slices):
      out_slices.append(self.expander_stack[i](slice.transpose(-1, -2)))
    return torch.squeeze(torch.stack(out_slices, dim=-1), dim=1)
  

# Lightning wrapper class for decoder only training
class LightningDecoder(L.LightningModule):
  def __init__(self, model):
    super().__init__()
    self.decoder=model
    self.mae_loss = nn.L1Loss()

  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.decoder(x)
    loss = F.mse_loss(y_hat, y)
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.decoder(x)
    val_loss = F.mse_loss(y_hat, y)
    self.log('val_loss', val_loss, sync_dist=True)

  def test_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.decoder(x)
    test_loss = F.mse_loss(y_hat, y)
    mae = self.mae_loss(y_hat, y)
    self.log('test_loss', test_loss)
    self.log('mae_loss', mae)
    return test_loss, mae

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.decoder.parameters(), lr=LR)
    return optimizer
  
  @classmethod
  def load_with_model(cls, checkpoint_path, model_class):
      model = model_class()
      wrapper = cls(model)
      checkpoint = torch.load(checkpoint_path)
      wrapper.load_state_dict(checkpoint['state_dict'])
      return wrapper
  