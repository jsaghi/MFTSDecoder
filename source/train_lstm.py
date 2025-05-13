from settings import *
import data
from lstm import LSTM_Predict, MFLSTM, LightningLSTM
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger


# Function that fits a LightningLSTM to a training dataset
def fit_lstm(model, name, train, val, num_epochs):
  lightning_lstm = LightningLSTM(model)

  early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=1e-5,
    mode='min',
    patience=5,
    verbose=True
  )
  logger = CSVLogger(save_dir=HISTORY_PATH + name)
  checkpoint = ModelCheckpoint(
      monitor='val_loss',
      dirpath=MODEL_PATH,
      filename=name + '-{epoch}',
      save_top_k=5,
      mode='min',
      verbose=True
    )
  trainer = L.Trainer(
      max_epochs=num_epochs,
      logger=logger,
      callbacks=[checkpoint, early_stopping]
    )
  
  trainer.fit(lightning_lstm, train, val)
  return 0

# Main training function. Builds a dictionary of models and datsets
# to train, then fits all of the models in the dictionary
def main():

  # Set torch.matmul precision
  torch.set_float32_matmul_precision('high')

  q1_train, q1_val, _ = data.get_mf_lstm_data(1)
  q2_train, q2_val, _ = data.get_mf_lstm_data(2)
  q3_train, q3_val, _ = data.get_mf_lstm_data(3)

  e = 100

  fit_dict = {
    'mf_lstm_q1': [MFLSTM(), q1_train, q1_val],
    'mf_lstm_q2': [MFLSTM(), q2_train, q2_val],
    'mf_lstm_q3': [MFLSTM(), q3_train, q3_val]
  }

  for key, value in fit_dict.items():
    fit_lstm(value[0], key, value[1], value[2], e)
  
  print('LSTM training complete')

if __name__ == '__main__':
  main()
