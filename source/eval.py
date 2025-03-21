import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from settings import *


# Function to parse raw CSV files, take the mean of training loss,
# and create a new dataframe which has average training loss and 
# val loss separated by epoch
def metrics_df(filename):
  raw_df = pd.read_csv(HISTORY_PATH + filename + '.csv')
  train_loss = []
  val_loss = []
  epoch = []
  num_epochs = raw_df['epoch'].nunique()
  for i in range(num_epochs):
    train_loss.append(raw_df[raw_df['epoch'] == i]['train_loss'].mean())
    val_loss.append(raw_df[raw_df['epoch'] == i]['val_loss'].mean())
    epoch.append(i)
  train_metrics = pd.DataFrame(
      {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss}
  )
  return train_metrics


# Function to pad a dataframe with NaN values if it isn't the same length
# as MAX_EPOCHS so that it can be plotted with other dataframes
def pad_df(data):
  pad_length = MAX_EPOCHS - len(data)
  for i in range(pad_length):
    new_row = pd.DataFrame({'epoch': [len(data) + i],
                            'train_loss': [np.nan],
                            'val_loss': [np.nan]})
    data = pd.concat([data, new_row], ignore_index=True)
  return data


# Code to graph val loss from a list of decoders against one another
def plot_val_loss(decoder_list):
  df_dict = {}
  for decoder in decoder_list:
    df_dict[decoder] = pad_df(metrics_df(decoder))
  x = np.arange(MAX_EPOCHS)
  for key, value in df_dict.items():
    plt.plot(x, value['val_loss'], label=key)
  plt.xlabel('Epoch')
  plt.ylabel('Validation Loss')
  plt.legend()
  plt.show()
