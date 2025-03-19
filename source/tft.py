from settings import *
import torch
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
import optim
import lightning as L


# Function to build a Temporal Fusion Transformer using the pytorch_forecasting
# class and an input dataset
def build_tft(dataset):
    tft = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=TFT_LR,
        hidden_size=HIDDEN_SIZE,
        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
        lstm_layers=LSTM_LAYERS,
        attention_head_size=ATTENTION_HEAD_SIZE,
        output_size=OUTPUT_SIZE,
        dropout=DROPOUT,
        loss=QuantileLoss(QUANTILES),
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
    y_hat = self.tft_model(x)[0]
    loss = loss_fn(y_hat, y)
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    loss_fn = self.tft_model.loss
    y_hat = self.tft_model(x)[0]
    loss = loss_fn(y_hat, y)
    self.log('val_loss', loss)

  def configure_optimizers(self):
    optimizer = optim.Ranger(self.tft_model.parameters(),
                       lr=TFT_LR,
                       weight_decay=WEIGHT_DECAY,
                       betas=(BETA1, BETA2),
                       eps=EPS,
                       k=K,
                       alpha=ALPHA,
                       N_sma_threshhold=5,)
    return optimizer
  