from settings import *
import data
import tft
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

training, train, val, _ = data.get_time_series()
base_tft = tft.build_tft(training)
lightning_tft = tft.LightningTFT(base_tft)

early_stopping = EarlyStopping(
  monitor='val_loss',
  min_delta=1e-5,
  mode='min',
  patience=5,
  verbose=True
  )
logger = CSVLogger(save_dir=HISTORY_PATH + 'tft_4')
checkpoint = ModelCheckpoint(
    monitor='val_loss',
    dirpath=MODEL_PATH,
    filename='tft_4' + '-{epoch}',
    save_top_k=5,
    mode='min',
    verbose=True
  )
trainer = L.Trainer(
    max_epochs=50,
    logger=logger,
    callbacks=[checkpoint, early_stopping]
)

trainer.fit(lightning_tft, train, val)