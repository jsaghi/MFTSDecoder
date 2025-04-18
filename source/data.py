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
    return len(self.lf_data) - (self.seq_length)
  
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


# Helper function that splits given datasets into train, test, and validate subsets
def train_val_test_split(data):
  validate_split = int((VAL_RATIO + TEST_RATIO) * len(data))
  test_split = int(TEST_RATIO * len(data))
  train = data[:-validate_split]
  val = data[-validate_split:-test_split]
  test = data[-test_split:]
  return train, val, test


# Function to build a pytorch_forecasting TimeSeriesDataSet for the TFT model from an input
# pre-scaled pandas dataframe of the jena climate data
def build_time_series(data):
  training_cutoff = data['time_idx'].max() - DELAY
  tsds = TimeSeriesDataSet(
  data[lambda x: x.time_idx <= training_cutoff],
  time_idx = 'time_idx',
  target = 'Targets',
  max_encoder_length = SEQ_LENGTH,
  max_prediction_length = DELAY,
  time_varying_known_categoricals = ['season'],
  time_varying_known_reals = ['month', 'hour_of_day'],
  time_varying_unknown_reals = [
      'T (degC)',
      'Tdew (degC)',
      'rh (%)',
      'sh (g/kg)',
      'H2OC (mmol/mol)',
      'p (mbar)',
      'VPmax (mbar)',
      'VPact (mbar)',
      'VPdef (mbar)',
      'rho (g/m**3)',
      'wv (m/s)',
      'max wv (m/s)',
      'wd (deg)'
  ],
  group_ids = ['group_id'],
  target_normalizer = None,
  add_relative_time_idx = False,
  add_target_scales = False,
  add_encoder_length = False,
  allow_missing_timesteps = True
  )
  return tsds


# Helper function to generate categorical season categories based on the month in a 
# pd.DataFrame of the jena climate dataset
def get_seasons(month):
  if month in [12, 1, 2]:
    return 'winter'
  elif month in [3, 4, 5]:
    return 'spring'
  elif month in [6, 7, 8]:
    return 'summer'
  else:
    return 'fall'
  

# Helper function to encode seasons to integers
def encode_seasons(season):
  if season == 'winter':
    return 0
  elif season == 'spring':
    return 1
  elif season == 'summer':
    return 2
  else:
    return 3


# Function to return normalized values of the jena climate dataset. Right-skewed features
# Have a modified logarithmic transform applied, and then min-max scaling is applied to all
# features except for temperature. This is because temperature will be the target, so 
# min-max scaling is applied to temperature inside of the relevant torch Dataset classes
def scale_jena():
  scaler = MinMaxScaler()
  jena = pd.read_csv(JENA_PATH)

  jena.drop(['Tpot (K)'], axis=1, inplace=True)
  jena['Date Time'] = pd.to_datetime(jena['Date Time'], format='%d.%m.%Y %H:%M:%S')
  group_id = pd.DataFrame(np.zeros(jena.shape[0], dtype=np.int32), columns=['group_id'])
  jena['season'] = jena['Date Time'].dt.month.apply(get_seasons)
  jena['month'] = jena['Date Time'].dt.month.astype(int)
  jena['hour_of_day'] = jena['Date Time'].dt.hour.astype(int)
  time_idx = np.arange(jena.shape[0])

  unscaled = pd.concat([pd.DataFrame(time_idx, columns=['time_idx']),
                      jena[['Date Time', 'season', 'month', 'hour_of_day']],
                      group_id], axis=1)
  targets = jena['T (degC)'].rename('Targets')
  log_transform = jena[['VPmax (mbar)',
                        'VPdef (mbar)',
                        'sh (g/kg)',
                        'H2OC (mmol/mol)',
                        'wv (m/s)',
                        'max. wv (m/s)']].apply(lambda x: np.log(x + 1))
  jena.drop(['Date Time',
             'season',
             'month',
             'hour_of_day',
             'VPmax (mbar)',
             'VPdef (mbar)',
             'sh (g/kg)',
             'H2OC (mmol/mol)',
             'wv (m/s)',
             'max. wv (m/s)'], axis=1, inplace=True)
  
  scaled_data = pd.concat([log_transform, jena], axis=1)
  scaled_df = pd.DataFrame(scaler.fit_transform(scaled_data), columns=scaled_data.columns)
  jena_scaled = pd.concat([unscaled, scaled_df, targets], axis=1)
  jena_scaled.rename(columns={'max. wv (m/s)': 'max wv (m/s)'}, inplace=True)
  return jena_scaled


# Function that returns a dataloader for train and validation datasets of just the 
# temperature dataset. Used to train decoder modules
def get_temp(downsample_ratio, scale_targets=True):
  jena = pd.read_csv(JENA_PATH)
  scaler = MinMaxScaler()
  scaled_temp = scaler.fit_transform(jena['T (degC)'].to_numpy().reshape(-1, 1))
  scaled_df = pd.DataFrame(scaled_temp.squeeze(), columns=['scaled'])
  lf_df = pd.concat([scaled_df, jena['T (degC)']], axis=1)
  train, val, test = train_val_test_split(lf_df.to_numpy())

  train_ds = LFData(train, SEQ_LENGTH, downsample_ratio, scale_targets)
  val_ds = LFData(val, SEQ_LENGTH, downsample_ratio, scale_targets)
  test_ds = LFData(test, SEQ_LENGTH, downsample_ratio, scale_targets)

  train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=5)
  val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=5)
  test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=5)
  return train_loader, val_loader, test_loader


# Function that returns a time series dataset for the jena climate dataset. Returns
# a dataloader for train, validation, and test datasets. Used to train prediction models without
# attached decoders
def get_time_series(downsample_ratio = None):
  jena_scaled = scale_jena()
  if downsample_ratio != None:
    jena_downsampled = jena_scaled.iloc[downsample_ratio - 1::downsample_ratio, :]
    jena_downsampled.drop(['time_idx'], axis=1, inplace=True)
    time_idx = np.arange(jena_downsampled.shape[0])
    jena_downsampled = pd.concat([pd.DataFrame(time_idx, columns=['time_idx']),
                                  jena_downsampled], axis=1
                                  )
    train, val, test = train_val_test_split(jena_downsampled)
  else:
    train, val, test = train_val_test_split(jena_scaled)
  training = build_time_series(train)
  validation = build_time_series(val)
  testing = build_time_series(test)
  train_loader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=11)
  val_loader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=11)
  test_loader = testing.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=11)
  return (training, train_loader, val_loader, test_loader)


# Function that returns train, validation, and test dataloaders for mixed frequency multivariate
# time series. This is the data pipline to train and test the combined prediction models
def get_mfts():
  data = scale_jena()
  data.drop(['Date Time'], axis=1, inplace=True)

  dataset = build_time_series(data)

  lf_data = data[['T (degC)', 'Tdew (degC)', 'rh (%)', 'sh (g/kg)', 'H2OC (mmol/mol)']].to_numpy()
  if_data = data[['p (mbar)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)']].to_numpy()
  hf_data = data[['rho (g/m**3)', 'wv (m/s)', 'max wv (m/s)', 'wd (deg)', 'Targets']].to_numpy() 
  data['season'] = data['season'].apply(lambda x: (encode_seasons(x)))
  kf_data = data[['time_idx', 'season', 'month', 'hour_of_day']].to_numpy()

  lf_train, lf_val, lf_test = train_val_test_split(lf_data)
  if_train, if_val, if_test = train_val_test_split(if_data)
  hf_train, hf_val, hf_test = train_val_test_split(hf_data)
  kf_train, kf_val, kf_test = train_val_test_split(kf_data)

  train_ds = MFTSData(lf_train, if_train, hf_train, kf_train)
  val_ds = MFTSData(lf_val, if_val, hf_val, kf_val)
  test_ds = MFTSData(lf_test, if_test, hf_test, kf_test)

  train_loader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle=True, num_workers=5)
  val_loader = DataLoader(val_ds, batch_size = BATCH_SIZE, shuffle=False, num_workers=5)
  test_loader = DataLoader(test_ds, batch_size = BATCH_SIZE, shuffle=False, num_workers=5)

  return dataset, train_loader, val_loader, test_loader
