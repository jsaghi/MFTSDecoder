import torch
import optuna
from functools import partial
import pickle
from settings import *
import data
import opt_func

# Set torch.matmul precision
torch.set_float32_matmul_precision('medium')

# Build training dataset, train loader, and val loader
training, train_loader, val_loader, _ = data.get_time_series()

'''
# Learning rate study
lr_objective = partial(
  opt_func.lr_objective,
  dataset=training,
  train_loader=train_loader,
  val_loader=val_loader
)
lr_study = optuna.create_study(direction='minimize', storage=STORAGE_URL)
lr_study.optimize(lr_objective, n_trials=20)

# Save the results
with open (STUDY_PATH + 'lr_trials.pkl', 'wb') as f:
  pickle.dump(lr_study.trials, f)
with open (STUDY_PATH + 'lr_trials_df.pkl', 'wb') as f:
  pickle.dump(lr_study.trials_dataframe(), f)
with open (STUDY_PATH + 'lr_best_params.pkl', 'wb') as f:
  pickle.dump(lr_study.best_params, f)

# Extract best learning rate results for the tft hyperparameter study
best_lr = lr_study.best_params['lr']
best_weight_decay = lr_study.best_params['weight_decay']
best_k = lr_study.best_params['k']
best_alpha = lr_study.best_params['alpha']
'''

# TFT hyperparameter study
tft_objective = partial(
  opt_func.tft_objective,
  lr=TFT_LR,
  weight_decay=WEIGHT_DECAY,
  k=K,
  alpha=ALPHA,
  train_loader=train_loader,
  val_loader=val_loader
)

tft_study = optuna.create_study(direction='minimize')
tft_study.optimize(tft_objective, n_trials=50)

# Save the results
with open (STUDY_PATH + 'tft_trials.pkl', 'wb') as f:
  pickle.dump(tft_study.trials, f)
with open (STUDY_PATH + 'tft_trials_df.pkl', 'wb') as f:
  pickle.dump(tft_study.trials_dataframe(), f)
with open (STUDY_PATH + 'tft_best_params.pkl', 'wb') as f:
  pickle.dump(tft_study.best_params, f)
