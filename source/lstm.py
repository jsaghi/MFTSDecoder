from settings import *
from custom_layers import DenseFilterExpansion, FilterAttention, ConvInsert
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class LSTM_Predict(nn.Module):
  def __init__(self, output_size):
    super().__init__()
    self.hidden_size = LSTM_HIDDEN_SIZE
    self.output_size = output_size

    self.encoder = nn.LSTM(LSTM_INPUT_SIZE,
                           self.hidden_size,
                           num_layers=NUM_LSTM_LAYERS,
                           batch_first=True,
                           dropout=LSTM_DROPOUT)
    self.decoder = nn.LSTM(1, self.hidden_size,
                           num_layers=NUM_LSTM_LAYERS,
                           batch_first=True,
                           dropout=LSTM_DROPOUT)
    self.projection = nn.Linear(self.hidden_size, 1)

  def forward(self, inputs):
    batch_size = inputs.size(0)

    _, (hidden, cell) = self.encoder(inputs)
    decoder_input = torch.zeros(batch_size, 1, 1, device=inputs.device)

    outputs = []
    for _ in range(self.output_size):
      out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
      step_output = self.projection(out)
      outputs.append(step_output)
      decoder_input = step_output

    return torch.cat(outputs, dim=1)
    
class LightningLSTM(L.LightningModule):
  def __init__(self, model):
    super().__init__()
    #self.save_hyperparameters()
    self.model = model
    self.loss = nn.MSELoss()

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = self.loss(y_hat, y)
    self.log("train_loss", loss, sync_dist=True)
    return loss
    
  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = self.loss(y_hat, y)
    self.log("val_loss", loss, sync_dist=True)
    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=LSTM_LR)
  
  @classmethod
  def load_with_model(cls, checkpoint_path, mf=False, delay=None):
      if not mf:
        model = LSTM_Predict(delay)
      else:
        model = MFLSTM()
      wrapper = cls(model)
      checkpoint = torch.load(checkpoint_path)
      wrapper.load_state_dict(checkpoint['state_dict'])
      return wrapper
  

# Decoder class to expand by a ratio of 6 to 1
class Expander6X(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.ConvTranspose1d(in_channels=1, out_channels=64, kernel_size=5,
                      stride=2, padding=2, output_padding=1)
    self.fa1 = FilterAttention(64, 336)
    self.ci1 = ConvInsert(336, 8)
    self.conv2 = nn.ConvTranspose1d(in_channels=1, out_channels=64,
                       kernel_size=3, stride=1, padding=0)
    self.conv3 = nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=5,
                      stride=2, padding=2, output_padding=1)
    self.fa2 = FilterAttention(128, SEQ_LENGTH)

  def forward(self, inputs):
    outputs = self.conv1(inputs)
    outputs = self.fa1(outputs)
    outputs = self.ci1(outputs)
    outputs = self.conv2(outputs)
    outputs = self.conv3(outputs)
    outputs = self.fa2(outputs)
    outputs = F.sigmoid(outputs)
    return outputs
    

# Variant using the new ConvInsert layer
class Expander36X(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.ConvTranspose1d(in_channels=1, out_channels=1024,
                       kernel_size=5, stride=2, padding=2, output_padding=1)
    self.conv2 = nn.ConvTranspose1d(in_channels=1024, out_channels=512,
                       kernel_size=5, stride=2, padding=2, output_padding=1)
    self.fa1 = FilterAttention(512, 112)
    self.ci1 = ConvInsert(112, 28)
    self.conv3 = nn.ConvTranspose1d(in_channels=1, out_channels=2048,
                       kernel_size=5, stride=2, padding=2, output_padding=1)
    self.conv4 = nn.ConvTranspose1d(in_channels=2048, out_channels=1024,
                       kernel_size=5, stride=2, padding=2, output_padding=1)
    self.conv5 = nn.ConvTranspose1d(in_channels=1024, out_channels=512,
                       kernel_size=5, stride=2, padding=2, output_padding=1)
    self.fa2 = FilterAttention(512, SEQ_LENGTH)

  def forward(self, inputs):
    outputs = self.conv1(inputs)
    outputs = self.conv2(outputs)
    outputs = self.fa1(outputs)
    outputs = self.ci1(outputs)
    outputs = self.conv3(outputs)
    outputs = self.conv4(outputs)
    outputs = self.conv5(outputs)
    outputs = self.fa2(outputs)
    outputs = F.sigmoid(outputs)
    return outputs
  

# Class that stacks 5 Expader36X modules, splits inputs along features, feeds
# each feature through a separate expander, and then stacks the outputs again
class LFExpanderStack(nn.Module):
  def __init__(self, num_features, out_length):
    super().__init__()
    self.expander_stack = nn.ModuleList([Expander36X() for _ in range(num_features)])

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
    self.expander_stack = nn.ModuleList([Expander6X() for _ in range(num_features)])

  def forward(self, inputs):
    in_slices = torch.split(inputs, 1, dim=-1)
    out_slices = []
    for i, slice in enumerate(in_slices):
      out_slices.append(self.expander_stack[i](slice.transpose(-1, -2)))
    return torch.squeeze(torch.stack(out_slices, dim=-1), dim=1)
  

# Class that combines convolutional upsampling with the 
class MFLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lf_stack = LFExpanderStack(NUM_LF_INPUTS, SEQ_LENGTH)
        self.if_stack = IFExpanderStack(NUM_IF_INPUTS, SEQ_LENGTH)
        self.lstm = LSTM_Predict(DELAY)
        self.loss = nn.MSELoss()

    def forward(self, inputs):
        # Separate inputs
        lf_in, if_in, hf_in = inputs

        # Upsample LF and IF inputs and concatenate them with known future variables and 
        # HF inputs
        lf_out = self.lf_stack(lf_in)
        if_out = self.if_stack(if_in)
        mf_real = torch.cat((lf_out, if_out, hf_in), axis=-1)

        outputs=self.lstm(mf_real)
        return outputs
