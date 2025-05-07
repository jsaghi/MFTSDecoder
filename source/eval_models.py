from settings import *
import data
from lstm import LSTM_Predict, MFLSTM, LightningLSTM
from tft import LightningTFT
from mftft import MFTFT, LightningMFTFT
import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger
import pickle

# Set torch.matmul precision
torch.set_float32_matmul_precision('high')
'''
# Build a dictionary of results to be saved after all evaluation has been completed
lstm_results = {}

# Evaluate all lstm models

# Generate test datasets
_, _, lf_lstm_test = data.get_lstm_ts(36)
_, _, hf_lstm_test = data.get_lstm_ts()
_, _, midas_lstm_test = data.get_lstm_ts_imputed()
_, _, mf_lstm_test = data.get_mf_lstm_data()

# Build a dictionary of models and their checkpoint filenames
lstm_models = {
  'lf_lstm' : ['lf_lstm-epoch=69.ckpt', False, LF_DELAY, lf_lstm_test],
  'hf_lstm' : ['hf_lstm-epoch=11.ckpt', False, DELAY, hf_lstm_test],
  'midas_lstm' : ['lstm_imputed-epoch=19.ckpt', False, DELAY, midas_lstm_test],
  'mf_lstm' : ['mf_lstm-epoch=12.ckpt', True, DELAY, mf_lstm_test]
}

for key, value in lstm_models.items():
  logger = CSVLogger(save_dir=EVAL_PATH + key)
  trainer = L.Trainer(logger=logger)
  eval_model = LightningLSTM.load_with_model(MODEL_PATH + value[0], value[1], value[2])
  eval_model.eval()
  result = trainer.test(eval_model, value[3])
  lstm_results[key] = {'mse': result[0]['test_loss'], 'mae': result[0]['mae_loss']}

# Save all lstm results to a pickle file
with open(EVAL_PATH + 'cu_eval_results.pkl', 'wb') as f:
  pickle.dump(lstm_results, f)

# Evaluate all tft models
'''

# Generate test datasets
lf_dataset, _, _, lf_tft_test = data.get_time_series(36)
full_dataset, _, _, hf_tft_test = data.get_time_series()
_, _, _, midas_tft_test = data.get_imputed_ts()
_, _, _, mftft_test = data.get_mfts()

# Build a dictionary of models and their checkpoint filenames
tft_models = {
  'lf_tft' : ['lf_tft-epoch=153.ckpt', lf_dataset, lf_tft_test],
  'hf_tft' : ['hf_tft-epoch=2.ckpt', full_dataset, hf_tft_test],
  'midas_tft' : ['midas_tft-epoch=4.ckpt', full_dataset, midas_tft_test],
  'mftft' : ['mftft2-epoch=39.ckpt', full_dataset, mftft_test]
}

for key, value in tft_models.items():
  logger = CSVLogger(save_dir=EVAL_PATH + key)
  trainer = L.Trainer(logger=logger)
  if key != 'mftft':
    eval_model = LightningTFT.load_with_model(MODEL_PATH + value[0], value[1])
  else:
    eval_model = LightningMFTFT.load_with_model(MODEL_PATH + value[0], value[1])
  eval_model.eval()
  result = trainer.test(eval_model, value[2])
