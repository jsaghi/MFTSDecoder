from settings import *
import torch
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
import lightning as L


# Function to build a Temporal Fusion Transformer using the pytorch_forecasting
# class and an input dataset
def build_tft(dataset):
    tft = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=TFT_LR,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTENTION_HEAD_SIZE,
        #output_size=OUTPUT_SIZE,
        dropout=DROPOUT,
        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
        loss=QuantileLoss(),
        log_interval=LOG_INTERVAL,
        optimizer=TFT_OPTIMIZER,
        reduce_on_plateau_patience=PATIENCE
    )
    return tft


# Lightning wrapper class for the TFT prediction model
class LightningTFT(L.LightningModule):
  def __init__(self, model):
    super().__init__()
    self.tft_model = model

  def training_step(self, batch, batch_idx):
    x, y = batch
    loss_fn = self.tft_model.loss
    y_hat = self.tft_model(x)['prediction']
    loss = loss_fn(y_hat, y)
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    loss_fn = self.tft_model.loss
    y_hat = self.tft_model(x)['prediction']
    loss = loss_fn(y_hat, y)
    self.log('val_loss', loss)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.tft_model.parameters(), lr=TFT_LR)
    return optimizer
  