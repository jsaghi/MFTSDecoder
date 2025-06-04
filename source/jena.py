from settings import *
import data
from data import LFData, MFTSData, LSTM_TS, MF_LSTM
import torch
from torch.utils.data import DataLoader
from pytorch_forecasting import TimeSeriesDataSet
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Class to group all of the functions used to preprocess and package data from the
# Jena cliamte dataset into the required format for training/evaluation
class Jena():
  # Init function handles importing the necessary data and scaling it (applying a logarthimic
  # transform if the data is right-skewed and then MinMax scaling to all features). The impute
  # flag determines whether IF/LF data should be imported from the imputed data file or if all
  # data should come from the original Jena dataset file. The init funciton generates 3 attributes:
  # self.targets, self.metadata, and self.scaled_data
  def __init__(self, impute=False):
    # Import the jena dataset, define targets, format the datetime, and rename max wv column
    raw_data = pd.read_csv(JENA_PATH)
    self.targets = raw_data['T (degC)'].rename('Targets')
    raw_data['Date Time'] = pd.to_datetime(raw_data['Date Time'], format='%d.%m.%Y %H:%M:%S')
    raw_data.rename(columns={'max. wv (m/s)': 'max wv (m/s)'}, inplace=True)

    # Generate the metadata (known future variables, time index, and group id) used by
    # the TFT model
    self.metadata = pd.DataFrame(np.arange(raw_data.shape[0], dtype=np.int32), columns=['time_idx'])
    self.metadata['group_id'] = pd.DataFrame(np.zeros(raw_data.shape[0], dtype=np.int32))
    self.metadata['season'] = raw_data['Date Time'].dt.month.apply(data.get_seasons)
    self.metadata['month'] = raw_data['Date Time'].dt.month.astype(int)
    self.metadata['hour_of_day'] = raw_data['Date Time'].dt.hour.astype(int)

    # Drop the temperature in K and datetime columns from the raw dataframe
    raw_data.drop(['Date Time', 'Tpot (K)'], axis=1, inplace=True)

    # If the flag for using imputed data is set to true, import the imputed data file
    # and concatenate it with the "high frequency" variables present in the original 
    # (non-imputed) dataset
    if impute:
      imputed_raw = pd.read_csv(IMPUTED_DATA_PATH)
      imputed_data = imputed_raw.iloc[:, 1:]
      hf_data = raw_data[['rho (g/m**3)', 'wv (m/s)', 'max wv (m/s)', 'wd (deg)']]
      jena_unscaled = pd.concat([imputed_data, hf_data], axis=1)

    # If the flag for using imputed data is set to false, create a reference to the 
    # original (raw) data for subsequent scaling operations
    else:
      jena_unscaled = raw_data

    # Initialize the scaler
    scaler = MinMaxScaler()

    # Apply a logarithmic transform to all right-skewed variables
    log_transform = jena_unscaled[['VPmax (mbar)',
                          'VPdef (mbar)',
                          'sh (g/kg)',
                          'H2OC (mmol/mol)',
                          'wv (m/s)',
                          'max wv (m/s)']].apply(lambda x: np.log(x + 1))
    
    # Drop the transformed columns from the unscaled dataframe
    jena_unscaled.drop(['VPmax (mbar)',
              'VPdef (mbar)',
              'sh (g/kg)',
              'H2OC (mmol/mol)',
              'wv (m/s)',
              'max wv (m/s)'], axis=1, inplace=True)
  
    # Concatenate the transformed and remaining data together again
    scaled_data = pd.concat([log_transform, jena_unscaled], axis=1)
    # Perform MinMax scaling on the data
    self.scaled_data = pd.DataFrame(scaler.fit_transform(scaled_data), columns=scaled_data.columns)


  # Function to build a pytorch_forecasting TimeSeriesDataSet for the TFT model using
  # the metadata, scaled data, and targets of the Jena object
  def build_time_series(dataset, delay=DELAY, seq_length=SEQ_LENGTH):
    training_cutoff = dataset['time_idx'].max() - delay
    tsds = TimeSeriesDataSet(
    dataset[lambda x: x.time_idx <= training_cutoff],
    time_idx = 'time_idx',
    target = 'Targets',
    max_encoder_length = seq_length,
    max_prediction_length = delay,
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


  # Function that returns a dataloader for train and validation datasets of just the 
  # temperature feature (target attribute) of the dataset. Used to train convolutional
  # upsampling modules
  def get_temp(self, downsample_ratio, scale_targets=True):
    lf_df = pd.concat([self.scaled_data['T (degC)'], self.targets()], axis=1)
    train, val, test = data.train_val_test_split(lf_df.to_numpy())

    train_ds = LFData(train, SEQ_LENGTH, downsample_ratio, scale_targets)
    val_ds = LFData(val, SEQ_LENGTH, downsample_ratio, scale_targets)
    test_ds = LFData(test, SEQ_LENGTH, downsample_ratio, scale_targets)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=5)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=5)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=5)
    return train_loader, val_loader, test_loader


  # Function to return a time series dataset for use with the LSTM prediction model. Contains
  # a downsample ratio parameter which allows the dataset to be downsampled so that a model
  # can be trained on either the full dataset or a low frequency version of it
  def get_lstm_ts(self, downsample_ratio = None):
    jena_scaled = pd.concat([self.scaled_data, self.targets], axis=1)

    # Downsample based on the downsampling ratio (if present)
    if downsample_ratio != None:   
      jena_downsampled = jena_scaled.iloc[downsample_ratio - 1::downsample_ratio, :]
      jena_downsampled.reset_index(drop=True, inplace=True)
      train, val, test = data.train_val_test_split(jena_downsampled)
      training = LSTM_TS(train, LF_LENGTH, LF_DELAY)
      validation = LSTM_TS(val, LF_LENGTH, LF_DELAY)
      testing = LSTM_TS(test, LF_LENGTH, LF_DELAY)
    
    # Bypass downsampling, use the full dataset
    else:
      train, val, test = data.train_val_test_split(jena_scaled)
      training = LSTM_TS(train, SEQ_LENGTH, DELAY)
      validation = LSTM_TS(val, SEQ_LENGTH, DELAY)
      testing = LSTM_TS(test, SEQ_LENGTH, DELAY)

    # Build dataloaders for training, validation, and testing
    train_loader = DataLoader(training, batch_size=BATCH_SIZE, shuffle=True, num_workers=11)
    val_loader = DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False, num_workers=11)
    test_loader = DataLoader(testing, batch_size=BATCH_SIZE, shuffle=False, num_workers=11)
    return (train_loader, val_loader, test_loader)



  # Function that returns a time series dataset for the jena climate dataset. Returns
  # a dataloader for train, validation, and test datasets, along with the full dataset
  # (which is needed to initialize a TFT model). Used to train TFT prediction
  # models without attached convolutional upsampling modules. While the time series
  # will all have the same sampling rate, the downsample ratio parameter allows for 
  # downsampling of all time series, meaning TFTs can be trained on either the full
  # dataset or a subsampled low frequency version of the full dataset
  def get_time_series(self, downsample_ratio=None):
    jena_scaled = pd.concat([self.metadata, self.scaled_data, self.targets], axis=1)

    # If the data is being downsampled, the time index needs to be dropped and then 
    # rebuilt after the dataset has been downsampled
    if downsample_ratio != None:
      jena_scaled.drop(['time_idx'], axis=1, inplace=True)
      jena_downsampled = jena_scaled.iloc[downsample_ratio - 1::downsample_ratio, :]
      jena_downsampled.reset_index(drop=True, inplace=True)
      time_idx = np.arange(jena_downsampled.shape[0])
      lf_jena = pd.concat([pd.DataFrame(time_idx, columns=['time_idx']),
                                    jena_downsampled], axis=1
                                    )
      train, val, test = data.train_val_test_split(lf_jena)
      training = self.build_time_series(train, LF_LENGTH, LF_DELAY)
      validation = self.build_time_series(val, LF_LENGTH, LF_DELAY)
      testing = self.build_time_series(test, LF_LENGTH, LF_DELAY)

    # No donwsampling, dataset can simply be split into training, validation, and test
    # sets and then passed to the build_time_series function
    else:
      train, val, test = data.train_val_test_split(jena_scaled)
      training = self.build_time_series(train)
      validation = self.build_time_series(val)
      testing = self.build_time_series(test)

    # Build dataloaders for the training, validation, and testing sets
    train_loader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=11)
    val_loader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=11)
    test_loader = testing.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=11)
    return (training, train_loader, val_loader, test_loader)



  # Function that returns train, validation, and test dataloaders for mixed frequency multivariate
  # time series. This is the data pipline to train and test the combined prediction models. Includes
  # a flag to indicate if the model to be trained is TFT or LSTM based (TFT is the default)
  def get_mfts(self, tft=True):
    jena = pd.concat([self.metadata, self.scaled_data, self.targets], axis=1)

    lf_data = jena[['T (degC)', 'Tdew (degC)', 'rh (%)', 'sh (g/kg)', 'H2OC (mmol/mol)']].to_numpy()
    if_data = jena[['p (mbar)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)']].to_numpy()
    hf_data = jena[['rho (g/m**3)', 'wv (m/s)', 'max wv (m/s)', 'wd (deg)', 'Targets']].to_numpy()

    lf_train, lf_val, lf_test = data.train_val_test_split(lf_data)
    if_train, if_val, if_test = data.train_val_test_split(if_data)
    hf_train, hf_val, hf_test = data.train_val_test_split(hf_data)

    if tft:
      dataset = self.build_time_series(jena)

      jena['season'] = jena['season'].apply(lambda x: (data.encode_seasons(x)))
      kf_data = jena[['time_idx', 'season', 'month', 'hour_of_day']].to_numpy()
      kf_train, kf_val, kf_test = data.train_val_test_split(kf_data)


      train_ds = MFTSData(lf_train, if_train, hf_train, kf_train)
      val_ds = MFTSData(lf_val, if_val, hf_val, kf_val)
      test_ds = MFTSData(lf_test, if_test, hf_test, kf_test)
    
    else:
      train_ds = MF_LSTM(lf_train, if_train, hf_train)
      val_ds = MF_LSTM(lf_val, if_val, hf_val)
      test_ds = MF_LSTM(lf_test, if_test, hf_test)

    train_loader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle=True, num_workers=5)
    val_loader = DataLoader(val_ds, batch_size = BATCH_SIZE, shuffle=False, num_workers=5)
    test_loader = DataLoader(test_ds, batch_size = BATCH_SIZE, shuffle=False, num_workers=5)

    if tft:
      return dataset, train_loader, val_loader, test_loader
    else:
      return train_loader, val_loader, test_loader
