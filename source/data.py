from settings import *

# LFData dataset class
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
  

class HFData(Dataset):
  def __init__(self, data, seq_length):
    self.data = data
    self.seq_length = seq_length
    self.mean = data.mean()
    self.std = data.std()

  def __len__(self):
    return len(self.data) - self.seq_length

  def __getitem__(self, index):
    tensor = torch.tensor(self.data[index:index + self.seq_length], dtype=torch.float32)
    tensor = tensor.unsqueeze(0)
    downsample -= self.mean
    downsample /= self.std
    return tensor, tensor

jena = pd.read_csv('data/jena_climate_2009_2016.csv')
tempc_df = jena['T (degC)']
tempc = tempc_df.to_numpy()
validate_split = int(VAL_RATIO * len(tempc))
train_data = tempc[:-validate_split]
val_data = tempc[-validate_split:]
downsample_size = MF_LENGTH
downsample_ratio = SEQ_LENGTH // downsample_size

train_ds = TempData(train_data, SEQ_LENGTH, downsample_ratio)
val_ds = TempData(val_data, SEQ_LENGTH, downsample_ratio)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)