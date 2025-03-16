from settings import *
import data
from expanders import LightningDecoder
from expander6x_variants import *
from expander36x_variants import *
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger


# Set torch.matmul precision
torch.set_float32_matmul_precision('medium')

# Build two dictionaries of models:
dict_6x = {
  '6xBase': Expander6XBase(),
  '6xHalfP': Expander6XHalfP(),
  '6xDoubleP': Expander6XDoubleP(),
  '6xDFE': Expander6XDFE(),
  '6xDFEHalfP': Expander6XHalfP(),
  '6xDFEDoubleP': Expander6XDFEDoubleP()
}

dict_36x = {
  #'36xConv': Expander36XConvOnly(),
  '36xFA': Expander36XFAOnly(),
  '36xDFE': Expander36XDFEFA(),
  '36xCIn': Expander36XConvIn(),
  '36xCInHalfP': Expander36XConvInHalfP(),
  '36xCInDoubleP': Expander36XConvInDoubleP(),
  '36xCInA': Expander36XConvInA(),
  '36xCInAHalfP': Expander36XConvInAHalfP(),
  '36xCInADoubleP': Expander36XConvInADoubleP()
}

# Generate training and validation dataloaders
lf_train_loader, lf_val_loader, _ = data.get_temp(SEQ_LENGTH // LF_LENGTH, False)
if_train_loader, if_val_loader, _ = data.get_temp(SEQ_LENGTH // IF_LENGTH, False)

# Prepare early stopping callback
early_stopping = EarlyStopping(
  monitor='val_loss',
  min_delta=1e-3,
  mode='min',
  patience=3,
  verbose=True
  )

# Train all models in the 6x dictionary
for key, value in dict_6x.items():
  logger = CSVLogger(save_dir=HISTORY_PATH + key)
  checkpoint = ModelCheckpoint(
    monitor='val_loss',
    dirpath=MODEL_PATH,
    filename=key + '-{epoch}',
    save_top_k=3,
    mode='min',
    verbose=True
  )
  trainer = L.Trainer(
    max_epochs=30,
    logger=logger,
    callbacks=[checkpoint, early_stopping],
  )

  trainer.fit(LightningDecoder(value), if_train_loader, if_val_loader)

# Train all models in the 36x dictionary
for key, value in dict_36x.items():
  logger = CSVLogger(HISTORY_PATH + key)
  checkpoint = ModelCheckpoint(
    monitor='val_loss',
    dirpath=MODEL_PATH,
    filename=key + '-{epoch}',
    save_top_k=3,
    mode='min',
    verbose=True
  )
  trainer = L.Trainer(
    max_epochs=30,
    logger=logger,
    callbacks=[checkpoint, early_stopping],
  )

  trainer.fit(LightningDecoder(value), lf_train_loader, lf_val_loader)
