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
trainer = L.Trainer(
    
)