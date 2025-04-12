from settings import *
from expanders import LFExpanderStack, IFExpanderStack
import tft
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
import torch_optimizer as optim
import lightning as L


class MFTFT(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.lf_stack = LFExpanderStack(NUM_LF_INPUTS)
        self.if_stack = IFExpanderStack(NUM_IF_INPUTS)
        self.tft = tft.build_tft(dataset)

    def forward(self, inputs):
        