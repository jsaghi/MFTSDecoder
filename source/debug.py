import lightning as L
from settings import *
import data
import opt_func
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning.loggers import CSVLogger
import torch


torch.set_float32_matmul_precision('medium')

# Build training dataset, train loader, and val loader
training, train_loader, val_loader, _ = data.get_time_series()

base_model = TemporalFusionTransformer(
  hidden_size=HIDDEN_SIZE,
  hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
  lstm_layers=LSTM_LAYERS,
  attention_head_size=ATTENTION_HEAD_SIZE,
  dropout=DROPOUT,
  output_size=OUTPUT_SIZE,
  loss=QuantileLoss(QUANTILES),
  log_interval=LOG_INTERVAL,
  reduce_on_plateau_patience=PATIENCE
)

model = opt_func.LTFTLRTuner(base_model, TFT_LR, WEIGHT_DECAY, K, ALPHA)
  
# Build trainer and train the model
trainer = L.Trainer(
  max_epochs=3,
  logger=CSVLogger(save_dir=HISTORY_PATH + 'tft_tuning'),
  callbacks=[EarlyStopping(monitor='val_loss', patience='5', mode='min')]
)

trainer.fit(model, train_loader, val_loader)