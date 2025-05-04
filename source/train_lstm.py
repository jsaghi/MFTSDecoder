from settings import *
import data
from lstm import LSTM_Predict, MFLSTM, LightningLSTM
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger


_, train, val, _ = data.get_mfts()
base_lstm = MFLSTM()
lightning_lstm = LightningLSTM(base_lstm)

early_stopping = EarlyStopping(
  monitor='val_loss',
  min_delta=1e-5,
  mode='min',
  patience=5,
  verbose=True
  )
logger = CSVLogger(save_dir=HISTORY_PATH + 'mf_lstm')
checkpoint = ModelCheckpoint(
    monitor='val_loss',
    dirpath=MODEL_PATH,
    filename='mf_lstm' + '-{epoch}',
    save_top_k=5,
    mode='min',
    verbose=True
  )
trainer = L.Trainer(
    max_epochs=100,
    logger=logger,
    callbacks=[checkpoint, early_stopping]
)

trainer.fit(lightning_lstm, train, val)
