import lightning as L
import torch
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
import torch_optimizer as optim
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning.loggers import CSVLogger
import pickle
from settings import *
import data
import tft


# Wrapper class for LightningTFT that allows for iteration on
# ranger optimizer hyperparameters
class LTFTLRTuner(L.LightningModule):
  def __init__(self, model, lr, weight_decay, k, alpha):
    super().__init__()
    self.tft = model
    self.lr = lr
    self.weight_decay = weight_decay
    self.k = k
    self.alpha = alpha
    self.automatic_optimization=False
  
  def training_step(self, batch, batch_idx):
    x, y = batch
    loss_fn = self.tft.loss
    y_hat = self.tft(x)[0]
    loss = loss_fn(y_hat, y)

    # Manual optimization
    self.manual_backward(loss)
    optimizer = self.optimizers()
    optimizer.step
    optimizer.zero_grad()

    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    loss_fn = self.tft.loss
    y_hat = self.tft(x)[0]
    loss = loss_fn(y_hat, y)
    self.log('val_loss', loss, on_epoch=True)

  def configure_optimizers(self):
    optimizer = optim.Ranger(self.tft.parameters(),
                       lr=self.lr,
                       weight_decay=self.weight_decay,
                       betas=(BETA1, BETA2),
                       eps=EPS,
                       N_sma_threshhold=5,
                       k=self.k,
                       alpha=self.alpha)
    return optimizer
  

# Trial to optimize hyperparameters for ranger optimizer
def lr_objective(trial, dataset, train_loader, val_loader):
  # Hyperparameters to iterate through
  lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
  weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
  k = trial.suggest_int('k', 5, 8)
  alpha = trial.suggest_uniform('alpha', 0.3, 0.6)

  # Build model and wrapped model
  base_model = tft.build_tft(dataset)
  model = LTFTLRTuner(base_model, lr, weight_decay, k, alpha)

  # Build trainer
  trainer = L.Trainer(
    max_epochs=20,
    logger=CSVLogger(save_dir=STUDY_PATH + 'lr_tuning'),
    callbacks=[
      EarlyStopping(monitor='val_loss', patience=3, mode='min'),
      PyTorchLightningPruningCallback(trial, monitor='val_loss')
    ]
  )

  # Train model
  trainer.fit(model, train_loader, val_loader)

  return trainer.callback_metrics.get('val_loss', torch.tensor(float('inf'))).item()


# Trial to optimize hyperparameters for the tft model
def tft_objective(trial, lr, weight_decay, k, alpha, train_loader, val_loader):
  # Hyperparameters to iterate through
  hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
  hidden_continuous_size = trial.suggest_categorical(
    'hidden_continuous_size', [32, 64, 128])
  attention_head_size = trial.suggest_int('attention_head_size', 2, 8)
  dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
  lstm_layers = trial.suggest_int('lstm_layers', 1, 3)
  
  # Build model and wrapped model
  base_model = TemporalFusionTransformer(
    hidden_size=hidden_size,
    hidden_continuous_size=hidden_continuous_size,
    lstm_layers=lstm_layers,
    attention_head_size=attention_head_size,
    dropout=dropout,
    output_size=OUTPUT_SIZE,
    loss=QuantileLoss(QUANTILES),
    log_interval=LOG_INTERVAL,
    reduce_on_plateau_patience=PATIENCE
  )

  model = LTFTLRTuner(base_model, lr, weight_decay, k, alpha)
  
  # Build trainer and train the model
  trainer = L.Trainer(
    max_epochs=30,
    logger=CSVLogger(save_dir=STUDY_PATH + 'tft_tuning'),
    #callbacks=[EarlyStopping(monitor='val_loss', patience='5', mode='min')]
  )

  trainer.fit(model, train_loader, val_loader)
  return trainer.callback_metrics.get('val_loss', torch.tensor(float('inf'))).item()
