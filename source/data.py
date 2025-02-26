from settings import *
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


# Downsampling dataset class for training and evaluating decoder models
class LFData(Dataset):
  def __init__(self, data, seq_length, downsample_ratio):
    self.data = data
    self.seq_length = seq_length
    self.downsample_ratio = downsample_ratio
    self.mean = data.mean(axis=0)
    self.std = data.std(axis=0)

  def __len__(self):
    return len(self.data) - self.seq_length

  def __getitem__(self, index):
    tensor = torch.tensor(self.data[index:index + self.seq_length], dtype=torch.float32)
    tensor = tensor.unsqueeze(0)
    downsample = tensor[::, self.downsample_ratio - 1::self.downsample_ratio]
    downsample -= self.mean
    downsample /= self.std
    return downsample, tensor
    

# Time series dataset class for evaluating predction models without decoders attached
class TSData(Dataset):
  def __init__(self, data, seq_length, delay):
    self.data = data
    self.targets = data[:, 1]
    self.seq_length = seq_length
    self.delay = delay
    self.mean = data.mean(axis=0)
    self.std = data.std(axis=0)

  def __len__(self):
    return len(self.data) - (self.seq_length + self.delay)
  
  def __getitem__(self, index):
    tensor = torch.tensor(self.data[index:index + self.seq_length], dtype=torch.float32)
    tensor -= self.mean
    tensor /= self.std
    target = torch.tensor(self.targets[index + self.seq_length + self.delay], dtype=torch.float32)
    return tensor, target


# Mixed Frequency dataset class for training combined models
class MFTSData(Dataset):
  def __init__(self, lf_data, if_data, hf_data, ds_ratio1, ds_ratio2, seq_length, delay):
    self.lf_data = lf_data
    self.lf_mean = lf_data.mean(axis=0)
    self.lf_std = lf_data.std(axis=0)

    self.if_data = if_data
    self.if_mean = if_data.mean(axis=0)
    self.if_std = if_data.std(axis=0)

    self.hf_data = hf_data
    self.hf_mean = hf_data.mean(axis=0)
    self.hf_std = hf_data.std(axis=0)

    self.targets = lf_data[:, 0]

    self.ds_ratio1 = ds_ratio1
    self.ds_ratio2 = ds_ratio2
    self.seq_length = seq_length
    self.delay = delay

  def __len__(self):
    return len(self.lf_data) - (self.seq_length + self.delay)
  
  def __getitem__(self, index):
    lf_tensor = torch.tensor(self.lf_data[index:index + self.seq_length], dtype=torch.float32)
    lf_tensor -= self.lf_mean
    lf_tensor /= self.lf_std

    if_tensor = torch.tensor(self.if_data[index:index + self.seq_length], dtype=torch.float32)
    if_tensor -= self.if_mean
    if_tensor /= self.if_std

    hf_tensor = torch.tensor(self.hf_data[index:index + self.seq_length], dtype=torch.float32)
    hf_tensor -= self.hf_mean
    hf_tensor /= self.hf_std

    target = torch.tensor(self.targets[index + self.seq_length + self.delay], dtype=torch.float32)

    lf_tensor = lf_tensor[self.ds_ratio1 - 1::self.ds_ratio1, :]
    if_tensor = if_tensor[self.ds_ratio2 - 1::self.ds_ratio2, :]

    lf_tensor.unsqueeze(0)
    if_tensor.unsqueeze(0)
    hf_tensor.unsqueeze(0)    

    return lf_tensor, if_tensor, hf_tensor, target


# Helper function that splits given datasets into train, test, and validate subsets
def train_val_test_split(data):
  validate_split = int((VAL_RATIO + TEST_RATIO) * len(data))
  test_split = int(TEST_RATIO * len(data))
  train = data[:-validate_split]
  val = data[-validate_split:-test_split]
  test = data[-test_split:]
  return train, val, test


# Function that returns a dataloader for train and validation datasets of just the 
# temperature dataset. Used to train decoder modules
def get_temp(downsample_ratio):
  jena = pd.read_csv(JENA_PATH)
  tempc_df = jena['T (degC)']
  train, val, test = train_val_test_split(tempc_df.to_numpy())

  train_ds = LFData(train, SEQ_LENGTH, downsample_ratio)
  val_ds = LFData(val, SEQ_LENGTH, downsample_ratio)
  test_ds = LFData(test, SEQ_LENGTH, downsample_ratio)

  train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
  val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
  return train_loader, val_loader, test_loader


# Function that returns a time series dataset for the jena climate dataset. Returns
# a dataloader for train, validation, and test datasets. Used to train prediction models without
# attached decoders
def get_ts():
  jena = pd.read_csv(JENA_PATH)
  jena.drop(['Tpot (K)', 'Date Time'], axis=1, inplace=True)
  train, val, test = train_val_test_split(jena.to_numpy())

  train_ds = TSData(train, SEQ_LENGTH, DELAY)
  val_ds = TSData(val, SEQ_LENGTH, DELAY)
  test_ds = TSData(test, SEQ_LENGTH, DELAY)

  train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
  val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

  return train_loader, val_loader, test_loader


# Function that returns train, validation, and test dataloaders for mixed frequency multivariate
# time series. This is the data pipline to train and test the combined prediction models
def get_mfts():
  jena = pd.read_csv(JENA_PATH)
  jena.drop(['Tpot (K)', 'Date Time'], axis=1, inplace=True)
  lf_data = jena[['T (degC)', 'Tdew (degC)', 'rh (%)', 'sh (g/kg)', 'H2OC (mmol/mol)']].to_numpy()
  if_data = jena[['p (mbar)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)']].to_numpy()
  hf_data = jena[['rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)']].to_numpy()

  lf_train, lf_val, lf_test = train_val_test_split(lf_data)
  if_train, if_val, if_test = train_val_test_split(if_data)
  hf_train, hf_val, hf_test = train_val_test_split(hf_data)

  train_ds = MFTSData(lf_train, if_train, hf_train, DOWNSAMPLE_RATIO1, DOWNSAMPLE_RATIO2, SEQ_LENGTH, DELAY)
  val_ds = MFTSData(lf_val, if_val, hf_val, DOWNSAMPLE_RATIO1, DOWNSAMPLE_RATIO2, SEQ_LENGTH, DELAY)
  test_ds = MFTSData(lf_test, if_test, hf_test, DOWNSAMPLE_RATIO1, DOWNSAMPLE_RATIO2, SEQ_LENGTH, DELAY)

  train_loader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle=True)
  val_loader = DataLoader(val_ds, batch_size = BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_ds, batch_size = BATCH_SIZE, shuffle=True)

  return train_loader, val_loader, test_loader
