from settings import *

# Downsampling dataset class for training and evaluating decoder models
class LFData(Dataset):
  def __init__(self, data, seq_length, downsample_ratio):
    self.data = data
    self.seq_length = seq_length
    self.downsample_ratio = downsample_ratio
    self.mean = data.mean()
    self.std = data.std()

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
    self.seq_length = seq_length
    self.delay = delay
    self.mean = data.mean()
    self.std = data.std()

  def __len__(self):
    return len(self.data) - (self.seq_length + self.delay)
  
  def __getitem__(self, index):
    tensor = torch.tensor(self.data[index:index + self.seq_length], dtype=torch.float32)
    target = torch.tensor(self.data[index + self.seq_length + self.delay])
    tensor -= self.mean
    tensor /= self.std
    return tensor, target

# Mixe Frequency dataset class for training combined models
class MFTSData(Dataset):
  def __init__(self, lf_data, if_data, hf_data, ds_ratio1, ds_ratio2, seq_length, delay):
    self.lf_data = lf_data
    self.if_data = if_data
    self.hf_data = hf_data
    self.ds_ratio1 = ds_ratio1
    self.ds_ratio2 = ds_ratio2
    self.seq_length = seq_length
    self.delay = delay
    self.mean = hf_data.mean()
    self.std = hf_data.std()

  def __len__(self):
    return len(self.data) - (self.seq_length + self.delay)
  
  def __getitem__(self, index):
    lf_tensor = torch.tensor(self.data[index:index + self.seq_length], dtype=torch.float32)
    if_tensor = torch.tensor(self.data[index:index + self.seq_length], dtype=torch.float32)
    hf_tensor = torch.tensor(self.data[index:index + self.seq_length], dtype=torch.float32)
    target = torch.tensor(self.data[index + self.seq_length + self.delay])

    lf_tensor -= self.mean
    lf_tensor /= self.std
    if_tensor -= self.mean
    if_tensor /= self.std
    hf_tensor -= self.mean
    hf_tensor /= self.std

    lf_tensor = lf_tensor[::, self.ds_ratio1 - 1::self.ds_ratio1]
    if_tensor = if_tensor[::, self.ds_ratio2 - 1::self.ds_ratio2]

    lf_tensor.unsqueeze(0)
    if_tensor.unsqueeze(0)
    hf_tensor.unsqueeze(0)    

    return lf_tensor, if_tensor, hf_tensor, target

# Function that returns a dataloader for train and validation datasets of just the 
# temperature dataset. Used to train decoder modules
def get_temp(downsample_ratio):
  jena = pd.read_csv(JENA_PATH)
  tempc_df = jena['T (degC)']
  tempc = tempc_df.to_numpy()
  validate_split = int(VAL_RATIO * len(tempc))
  train_data = tempc[:-validate_split]
  val_data = tempc[-validate_split:]

  train_ds = LFData(train_data, SEQ_LENGTH, downsample_ratio)
  val_ds = LFData(val_data, SEQ_LENGTH, downsample_ratio)

  train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
  val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)
  return train_loader, val_loader

# Function that returns a time series dataset for the jena climate dataset. Returns
# a dataloader for train, validation, and test datasets. Used to train prediction models without
# attached decoders
def get_ts():
  jena = pd.read_csv(JENA_PATH)
  jena = jena.drop(['Tpot (K)', 'Date Time'], inplace=True)
  jenanp = jena.to_numpy()
  validate_split = int((VAL_RATIO + TEST_RATIO) * len(jenanp))
  test_split = int(TEST_RATIO * len(jenanp))
  train_data = jenanp[:-validate_split]
  val_data = jenanp[-validate_split:-test_split]
  test_data = jenanp[-test_split:]

  train_ds = TSData(train_data, SEQ_LENGTH, DELAY)
  val_ds = TSData(val_data, SEQ_LENGTH, DELAY)
  test_ds = TSData(test_data, SEQ_LENGTH, DELAY)

  train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
  val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

  return train_loader, val_loader, test_loader

# Function that returns train, validation, and test dataloaders for mixed frequency multivariate
# time series. This is the data pipline to train and test the combined prediction models
def get_mfts():
  jena = pd.read_csv(JENA_PATH)
  jena = jena.drop(['Tpot (K)', 'Date Time'], inplace=True)
  lf_data = jena['T deg(C)', 'Tdew (degC)', 'rh (%)', 'sh (g/kg)', 'H2OC (mmol/mol)'].to_numpy()
  if_data = jena['p (mbar)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)'].to_numpy()
  hf_data = jena['rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)'].to_numpy()

  validate_split = int((VAL_RATIO + TEST_RATIO) * len(lf_data))
  test_split = int(TEST_RATIO * len(lf_data))
  lf_train = lf_data[:-validate_split]
  if_train = if_data[:-validate_split]
  hf_train = hf_data[:-validate_split]
  lf_val = lf_data[-validate_split:-test_split]
  if_val = if_data[-validate_split:-test_split]
  hf_val = hf_data[-validate_split:-test_split]
  lf_test = lf_data[-test_split:]
  if_test = lf_data[-test_split:]
  hf_test = lf_data[-test_split:]

  train_ds = MFTSData(lf_train, if_train, hf_train, DOWNSAMPLE_RATIO1, DOWNSAMPLE_RATIO2, SEQ_LENGTH, DELAY)
  val_ds = MFTSData(lf_val, if_val, hf_val, DOWNSAMPLE_RATIO1, DOWNSAMPLE_RATIO2, SEQ_LENGTH, DELAY)
  test_ds = MFTSData(lf_test, if_test, hf_test, DOWNSAMPLE_RATIO1, DOWNSAMPLE_RATIO2, SEQ_LENGTH, DELAY)

  train_loader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle=True)
  val_loader = DataLoader(val_ds, batch_size = BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_ds, batch_size = BATCH_SIZE, shuffle=True)

  return train_loader, val_loader, test_loader