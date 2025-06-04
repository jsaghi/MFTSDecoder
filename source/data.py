from settings import *
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_forecasting import TimeSeriesDataSet
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Downsampling dataset class for training and evaluating decoder models. Scales both inputs
# and targets using the MinMaxScaler. Outputs have a shape of:
# ((1, seq_length / downsample_ratio), (1, seq_length))
class LFData(Dataset):
  def __init__(self, data, seq_length, downsample_ratio, scale_targets):
    self.scaled = data[:, 0]
    self.targets = data[:, 1]
    self.seq_length = seq_length
    self.downsample_ratio = downsample_ratio
    self.scale_targets = scale_targets

  def __len__(self):
    return len(self.targets) - self.seq_length

  def __getitem__(self, index):
    scaled_tensor = torch.tensor(self.scaled.reshape(1, -1)[:, index:index + self.seq_length], dtype=torch.float32)
    downsample = scaled_tensor[:, self.downsample_ratio - 1::self.downsample_ratio]
    raw_tensor = torch.tensor(self.targets.reshape(1, -1)[:, index:index + self.seq_length], dtype=torch.float32)
    if self.scale_targets:
      return downsample, scaled_tensor
    return downsample, raw_tensor


# Mixed Frequency dataset class for training the MFTFT
class MFTSData(Dataset):
  def __init__(self, lf_data, if_data, hf_data, kf_data):
    self.lf_data = lf_data
    self.if_data = if_data
    self.hf_data = hf_data[:, :-1]
    self.targets = hf_data[:, -1]
    self.kf_data = kf_data

    self.ds_ratio1 = MFTFT_SEQ // MFTFT_LF
    self.ds_ratio2 = MFTFT_SEQ // MFTFT_IF
    self.seq_length = MFTFT_SEQ

  def __len__(self):
    return len(self.lf_data) - self.seq_length
  
  def __getitem__(self, index):
    lf_tensor = torch.tensor(self.lf_data[index:index + self.seq_length], dtype=torch.float32)
    if_tensor = torch.tensor(self.if_data[index:index + self.seq_length], dtype=torch.float32)
    hf_tensor = torch.tensor(self.hf_data[index:index + self.seq_length], dtype=torch.float32)
    kf_tensor = torch.tensor(self.kf_data[index:index + self.seq_length], dtype=torch.int64)
    target_tensor = torch.tensor(self.targets[index:index + self.seq_length], dtype=torch.float32)
    target = torch.tensor(self.targets[index + SEQ_LENGTH:index + self.seq_length], dtype=torch.float32)

    lf_tensor = lf_tensor[self.ds_ratio1 - 1::self.ds_ratio1, :]
    if_tensor = if_tensor[self.ds_ratio2 - 1::self.ds_ratio2, :]
 
    return (lf_tensor, if_tensor, hf_tensor, kf_tensor, target_tensor), target
  

# Dataset class for training the LSTM prediction model
class LSTM_TS(Dataset):
  def __init__(self, data, seq_length, delay):
    self.scaled = data.iloc[:, :-1].to_numpy()
    self.targets = data.iloc[:, -1:].to_numpy()
    self.seq_length = seq_length
    self.delay = delay

  def __len__(self):
    return len(self.targets) - (self.seq_length + self.delay)

  def __getitem__(self, index):
    x = torch.tensor(self.scaled[index:index + self.seq_length, :], dtype=torch.float32)
    y = torch.tensor(self.targets[index + self.seq_length:index + self.seq_length + self.delay, :], dtype=torch.float32)
    return x, y


# Dataset class for training the MF_LSTM model. Unlike the MFTFT, the MF_LSTM
# doesn't use future values during training, so the sequence length of data
# includes only the historical window, not the historical window + prediction horizon
class MF_LSTM(Dataset):
  def __init__(self, lf_data, if_data, hf_data):
    self.lf_data = lf_data
    self.if_data = if_data
    self.hf_data = hf_data[:, :-1]
    self.targets = hf_data[:, -1]

    self.ds_ratio1 = SEQ_LENGTH // LF_LENGTH
    self.ds_ratio2 = SEQ_LENGTH // IF_LENGTH
    self.seq_length = SEQ_LENGTH

  def __len__(self):
    return len(self.lf_data) - (self.seq_length + DELAY)
  
  def __getitem__(self, index):
    lf_tensor = torch.tensor(self.lf_data[index:index + self.seq_length], dtype=torch.float32)
    if_tensor = torch.tensor(self.if_data[index:index + self.seq_length], dtype=torch.float32)
    hf_tensor = torch.tensor(self.hf_data[index:index + self.seq_length], dtype=torch.float32)
    target = torch.tensor(
      self.targets[index + self.seq_length:index + self.seq_length + DELAY], dtype=torch.float32
      ).unsqueeze(-1)

    lf_tensor = lf_tensor[self.ds_ratio1 - 1::self.ds_ratio1, :]
    if_tensor = if_tensor[self.ds_ratio2 - 1::self.ds_ratio2, :]
 
    return (lf_tensor, if_tensor, hf_tensor), target


# Helper function that splits given datasets into train, test, and validate subsets
def train_val_test_split(data):
  validate_split = int((VAL_RATIO + TEST_RATIO) * len(data))
  test_split = int(TEST_RATIO * len(data))
  train = data[:-validate_split]
  val = data[-validate_split:-test_split]
  test = data[-test_split:]
  return train, val, test


  # Helper function to generate categorical season categories based on the month in a 
  # pd.DataFrame of the jena climate dataset
  def get_seasons(self, month):
    if month in [12, 1, 2]:
      return 'winter'
    elif month in [3, 4, 5]:
      return 'spring'
    elif month in [6, 7, 8]:
      return 'summer'
    else:
      return 'fall'
  

  # Helper function to encode seasons to integers
  def encode_seasons(self, season):
    if season == 'winter':
      return 0
    elif season == 'spring':
      return 1
    elif season == 'summer':
      return 2
    else:
      return 3
    