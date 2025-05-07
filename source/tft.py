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
    self.logging_metrics = {
       'mae' : MeanAbsoluteError(),
       'mse' : MeanSquaredError(),
    }

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

  def on_test_epoch_start(self):
    self.test_outputs = []

  def test_step(self, batch, batch_idx):
    x, y = batch
    loss_fn = self.tft_model.loss
    y_hat = self.tft_model(x)
    loss = loss_fn(y_hat, y)

    # Use median quantile for point estimate
    y_median = y_hat[:, :, 1]

    mae = self.logging_metrics['mae'](y_median, y)
    mse = self.logging_metrics['mse'](y_median, y)

    self.test_outputs.append({
      'loss': loss.detach(),
      'mae': mae.detach(),
      'mse': mse.detach()
    })

    return {'loss': loss, 'mae': mae, 'mse': mse}

  def on_test_epoch_end(self):
    avg_loss = torch.stack([x['loss'] for x in self.test_outputs]).mean()
    avg_mae = torch.stack([x['mae'] for x in self.test_outputs]).mean()
    avg_mse = torch.stack([x['mse'] for x in self.test_outputs]).mean()

    self.log('test_loss', avg_loss)
    self.log('mae_loss', avg_mae)
    self.log('mse_loss', avg_mse)

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
  