from settings import *
import torch
import torch.nn as nn
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
    self.save_hyperparameters()
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
    