from settings import *
import data
from expanders import LightningDecoder
from expander6x_variants import *
from expander36x_variants import *
import lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


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
  '36xConv': Expander36XConvOnly(),
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

# Train all models in the 6x dictionary
for key, value in dict_6x:
  logger = TensorBoardLogger(HISTORY_PATH + key)
  checkpoint_callback = ModelCheckpoint(
    moniotr='val_loss',
    dirpath=MODEL_PATH,
    filename=f'{key}-model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
    verbose=True
  )
  trainer = L.trainer(
    max_epochs=20,
    logger=logger,
    callbacks=[checkpoint_callback],
  )

  trainer.fit(value, if_train_loader, if_val_loader)

# Train all models in the 36x dictionary
for key, value in dict_36x:
  logger = TensorBoardLogger(HISTORY_PATH + key)
  checkpoint_callback = ModelCheckpoint(
    moniotr='val_loss',
    dirpath=MODEL_PATH,
    filename=f'{key}-model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
    verbose=True
  )
  trainer = L.trainer(
    max_epochs=20,
    logger=logger,
    callbacks=[checkpoint_callback],
  )

  trainer.fit(value, if_train_loader, if_val_loader)
  