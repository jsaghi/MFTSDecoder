from settings import *
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Downsampling dataset class for training and evaluating decoder models. Scales both inputs
# and targets using the MinMaxScaler. Outputs have a shape of:
# ((1, seq_length / downsample_ratio), (1, seq_length))
class LFData(Dataset):
  def __init__(self, data, seq_length, downsample_ratio):
    self.data = data
    self.seq_length = seq_length
    self.downsample_ratio = downsample_ratio
    self.scaler = MinMaxScaler()

  def __len__(self):
    return len(self.data) - self.seq_length

  def __getitem__(self, index):
    scaled_data = self.scaler.fit_transform(self.data.reshape(-1, 1))
    tensor = torch.tensor(scaled_data[index:index + self.seq_length], dtype=torch.float32)
    tensor = tensor.transpose(0, 1)
    downsample = tensor[::, self.downsample_ratio - 1::self.downsample_ratio]
    return downsample, tensor
    

# Time series dataset class for evaluating predction models without decoders attached
class TSData(Dataset):
  def __init__(self, data, seq_length, delay):
    self.data = data[:, :-1]
    self.targets = data[:, -1]
    self.seq_length = seq_length
    self.delay = delay

  def __len__(self):
    return len(self.data) - (self.seq_length + self.delay)
  
  def __getitem__(self, index):
    tensor = torch.tensor(self.data[index:index + self.seq_length], dtype=torch.float32)
    target = torch.tensor(self.targets[index + self.seq_length + self.delay], dtype=torch.float32)
    return tensor, target


# Mixed Frequency dataset class for training combined models
class MFTSData(Dataset):
  def __init__(self, lf_data, if_data, hf_data, ds_ratio1, ds_ratio2, seq_length, delay):
    self.lf_data = lf_data
    self.if_data = if_data
    self.hf_data = hf_data[:, :-1]
    self.targets = hf_data[:, -1]

    self.ds_ratio1 = ds_ratio1
    self.ds_ratio2 = ds_ratio2
    self.seq_length = seq_length
    self.delay = delay

  def __len__(self):
    return len(self.lf_data) - (self.seq_length + self.delay)
  
  def __getitem__(self, index):
    lf_tensor = torch.tensor(self.lf_data[index:index + self.seq_length], dtype=torch.float32)
    if_tensor = torch.tensor(self.if_data[index:index + self.seq_length], dtype=torch.float32)
    hf_tensor = torch.tensor(self.hf_data[index:index + self.seq_length], dtype=torch.float32)
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


# Function to return normalized values of the jena climate dataset. Right-skewed features
# Have a modified logarithmic transform applied, and then min-max scaling is applied to all
# features except for temperature. This is because temperature will be the target, so 
# min-max scaling is applied to temperature inside of the relevant torch Dataset classes
def jena_normalize():
  jena = pd.read_csv(JENA_PATH)
  jena.drop(['Tpot (K)'], axis=1, inplace=True)
  date_time = jena['Date Time']
  targets = jena['T (degC)'].rename('Targets')
  log_transform = jena[['VPmax (mbar)',
                        'VPdef (mbar)',
                        'sh (g/kg)',
                        'H2OC (mmol/mol)',
                        'wv (m/s)',
                        'max. wv (m/s)']].apply(lambda x: np.log(x + 1))
  jena.drop(['Date Time',
             'VPmax (mbar)',
             'VPdef (mbar)',
             'sh (g/kg)',
             'H2OC (mmol/mol)',
             'wv (m/s)',
             'max. wv (m/s)'], axis=1, inplace=True)
  
  scaler = MinMaxScaler()
  logt_scaled = pd.DataFrame(scaler.fit_transform(log_transform),
                             columns=log_transform.columns)
  jena_scaled = pd.DataFrame(scaler.fit_transform(jena), columns=jena.columns)
  output_df = pd.concat([date_time, jena_scaled, logt_scaled, targets], axis=1)
  return output_df


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
  data = jena_normalize()
  data.drop(['Date Time'], axis=1, inplace=True)
  train, val, test = train_val_test_split(data.to_numpy())

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
  data = jena_normalize()
  data.drop(['Date Time'], axis=1, inplace=True)
  lf_data = data[['T (degC)', 'Tdew (degC)', 'rh (%)', 'sh (g/kg)', 'H2OC (mmol/mol)']].to_numpy()
  if_data = data[['p (mbar)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)']].to_numpy()
  hf_data = data[['rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)', 'Targets']].to_numpy()

  lf_train, lf_val, lf_test = train_val_test_split(lf_data)
  if_train, if_val, if_test = train_val_test_split(if_data)
  hf_train, hf_val, hf_test = train_val_test_split(hf_data)

  train_ds = MFTSData(lf_train, if_train, hf_train, (SEQ_LENGTH // LF_LENGTH),
                      (SEQ_LENGTH // IF_LENGTH), SEQ_LENGTH, DELAY)
  val_ds = MFTSData(lf_val, if_val, hf_val, (SEQ_LENGTH // LF_LENGTH),
                    (SEQ_LENGTH // IF_LENGTH), SEQ_LENGTH, DELAY)
  test_ds = MFTSData(lf_test, if_test, hf_test, (SEQ_LENGTH // LF_LENGTH),
                     (SEQ_LENGTH // IF_LENGTH), SEQ_LENGTH, DELAY)

  train_loader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle=True)
  val_loader = DataLoader(val_ds, batch_size = BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_ds, batch_size = BATCH_SIZE, shuffle=True)

  return train_loader, val_loader, test_loader
