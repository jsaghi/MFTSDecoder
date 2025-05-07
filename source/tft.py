from settings import *
import torch
import torch.nn as nn
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
import torch_optimizer as optim
from torchmetrics import MeanAbsoluteError, MeanSquaredError
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
    self.automatic_optimization=False
    self.mae = MeanAbsoluteError()
    self.mse = MeanSquaredError()

  def training_step(self, batch, batch_idx):
    x, y = batch
    loss_fn = self.tft_model.loss
    y_hat = self.tft_model(x)[0]
    loss = loss_fn(y_hat, y)

    # Manual optimization
    self.manual_backward(loss)
    optimizer = self.optimizers()
    optimizer.step()
    optimizer.zero_grad()

    self.log('train_loss', loss, sync_dist=True)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    loss_fn = self.tft_model.loss
    y_hat = self.tft_model(x)[0]
    loss = loss_fn(y_hat, y)
    self.log('val_loss', loss, sync_dist=True)

  def test_step(self, batch, batch_idx):
    output = self.model(batch)
    y_hat = output['prediction']
    y = batch[0]['target']

    # Select 0.5 quantile (median)
    median_idx = self.model.quantiles.index(0.5)
    y_hat_median = y_hat[..., median_idx]

    self.mae.update(y_hat_median.flatten(), y.flatten())
    self.mse.update(y_hat_median.flatten(), y.flatten())

    return {'y': y, 'y_hat': y_hat_median}

  def test_epoch_end(self, outputs):
    mae = self.mae.compute()
    mse = self.mse.compute()
    self.log('test_mae', mae)
    self.log('test_mse', mse)
    self.mae.reset()
    self.mse.reset()

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

  @classmethod
  def load_with_model(cls, checkpoint_path, dataset):
      model = build_tft(dataset)
      wrapper = cls(model)
      checkpoint = torch.load(checkpoint_path)
      wrapper.load_state_dict(checkpoint['state_dict'])
      return wrapper
  