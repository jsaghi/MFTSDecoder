from settings import *
import data
from mftft import MFTFT, LightningMFTFT
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

# Set torch.matmul precision
torch.set_float32_matmul_precision('medium')

time_series, train, val, _ = data.get_mfts()

base_mftft = MFTFT(time_series)
lightning_mftft = LightningMFTFT(base_mftft)

'''
lightning_mftft = LightningMFTFT.load_with_model(
  MODEL_PATH + 'mftft-epoch=49.ckpt',
  MFTFT,
  time_series
  )
'''

early_stopping = EarlyStopping(
  monitor='val_loss',
  min_delta=1e-5,
  mode='min',
  patience=5,
  verbose=True
  )

logger = CSVLogger(save_dir=HISTORY_PATH + 'mftft_v2')
checkpoint = ModelCheckpoint(
    monitor='val_loss',
    dirpath=MODEL_PATH,
    filename='mftft_v2' + '-{epoch}',
    save_top_k=5,
    mode='min',
    verbose=True
  )

trainer = L.Trainer(
    max_epochs=150,
    logger=logger,
    callbacks=[checkpoint, early_stopping],
)

trainer.fit(lightning_mftft, train, val)
