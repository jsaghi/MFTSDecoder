from settings import *
from conv_upsamplers import LFExpanderStack, IFExpanderStack
import tft
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_forecasting import QuantileLoss
import torch_optimizer as optim
from torchmetrics import MeanAbsoluteError, MeanSquaredError
import lightning as L


class MFTFT(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.lf_stack = LFExpanderStack(NUM_LF_INPUTS, MFTFT_SEQ)
        self.if_stack = IFExpanderStack(NUM_IF_INPUTS, MFTFT_SEQ)
        self.tft = tft.build_tft(dataset)
        self.loss = QuantileLoss(QUANTILES)

    def forward(self, inputs):
        # Separate inputs
        lf_in, if_in, hf_in, kf_in, targets = inputs
        time_idx = kf_in[:, SEQ_LENGTH:]
        kf_cat = kf_in[:, :, 1:2]
        kf_real = kf_in[:, :, 2:]

        # Find Batch Size:
        batch = targets.size(0)

        # Find the device being used
        device = hf_in.device

        # Upsample LF and IF inputs and concatenate them with known future variables and 
        # HF inputs
        lf_out = self.lf_stack(lf_in)
        if_out = self.if_stack(if_in)
        mf_real = torch.cat((kf_real, lf_out, if_out, hf_in), axis=-1)

        # Split the data into encoder and decoder portions (where the encoder processes 
        # the historical values and the decoder processes values in the forecasting horizon)
        mf_encode = mf_real[:, :SEQ_LENGTH, :]
        mf_decode = mf_real[:, SEQ_LENGTH:, :]
        target_encode = targets[:, :SEQ_LENGTH]
        target_decode = targets[:, SEQ_LENGTH:]
        cat_encode = kf_cat[:, :SEQ_LENGTH, :]
        cat_decode = kf_cat[:, SEQ_LENGTH:, :]

        # Build a dictionary mimicicing a pytorch_forecasting TimeSeriesDataSet to feed to
        # the tft prediction module
        tft_in = {
            'encoder_cat': cat_encode,
            'encoder_cont': mf_encode,
            'encoder_target': target_encode,
            'encoder_lengths': torch.tensor(([SEQ_LENGTH] * batch),
                                            dtype=torch.int64, device=device),
            'decoder_cat': cat_decode,
            'decoder_cont': mf_decode,
            'decoder_target': target_decode,
            'decoder_lengths': torch.tensor(([DELAY] * batch),
                                            dtype=torch.int64, device=device),
            'decoder_time_idx': time_idx,
            'groups': torch.tensor(([0] * batch),
                                   dtype=torch.int64, device=device).unsqueeze(-1),
            'target_scale': torch.tensor(([[0, 1]] * batch),
                                         dtype=torch.float32, device=device)
        }

        outputs=self.tft(tft_in)
        return outputs


# Lightning wrapper class for the MFTFT prediction model
class LightningMFTFT(L.LightningModule):
  def __init__(self, model):
    super().__init__()
    self.mftft_model = model
    self.automatic_optimization=False
    self.logging_metrics = {
       'mae' : MeanAbsoluteError(),
       'mse' : MeanSquaredError(),
    }

  def training_step(self, batch, batch_idx):
    x, y = batch
    loss_fn = self.mftft_model.loss
    y_hat = self.mftft_model(x)[0]
    loss = loss_fn(y_hat, y)

    # Manual optimization
    self.manual_backward(loss)
    optimizer = self.optimizers()
    optimizer.step()
    optimizer.zero_grad()

    self.log('train_loss', loss, sync_dist=True)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    loss_fn = self.mftft_model.loss
    y_hat = self.mftft_model(x)[0]
    loss = loss_fn(y_hat, y)
    self.log('val_loss', loss, sync_dist=True)

  def on_test_epoch_start(self):
    self.test_outputs = []

  def test_step(self, batch, batch_idx):
    x, y = batch
    loss_fn = self.mftft_model.loss
    y_hat = self.mftft_model(x)[0]

    loss = loss_fn(y_hat, y)

    # Use median quantile for point estimate
    y_median = y_hat[:, :, 1]

    mae = self.logging_metrics['mae'](y_median, y)
    mse = self.logging_metrics['mse'](y_median, y)

    self.test_outputs.append({
      'loss': loss.detach(),
      'mae': mae.detach(),
      'mse': mse.detach()
    })

    return {'loss': loss, 'mae': mae, 'mse': mse}

  def on_test_epoch_end(self):
    avg_loss = torch.stack([x['loss'] for x in self.test_outputs]).mean()
    avg_mae = torch.stack([x['mae'] for x in self.test_outputs]).mean()
    avg_mse = torch.stack([x['mse'] for x in self.test_outputs]).mean()

    self.log('test_loss', avg_loss)
    self.log('mae_loss', avg_mae)
    self.log('mse_loss', avg_mse)

  def configure_optimizers(self):
    optimizer = optim.Ranger(self.mftft_model.parameters(),
                       lr=TFT_LR,
                       weight_decay=WEIGHT_DECAY,
                       betas=(BETA1, BETA2),
                       eps=EPS,
                       k=K,
                       alpha=ALPHA,
                       N_sma_threshhold=5,)
    return optimizer
  
  @classmethod
  def load_with_model(cls, checkpoint_path, dataset):
      model = MFTFT(dataset)
      wrapper = cls(model)
      checkpoint = torch.load(checkpoint_path)
      wrapper.load_state_dict(checkpoint['state_dict'])
      return wrapper
