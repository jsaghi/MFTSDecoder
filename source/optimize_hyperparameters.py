import optuna
from functools import partial
import pickle
from settings import *
import data
import opt_func

training, train_loader, val_loader, _ = data.get_time_series()

lr_objective = partial(
  opt_func.lr_objective,
  dataset=training,
  train_loader=train_loader,
  val_loader=val_loader
)

lr_study = optuna.create_study(direction='minimize', storage=STORAGE_URL)
lr_study.optimize(lr_objective, n_trials=20)

with open (STUDY_PATH + 'lr_study.pkl', 'wb') as f:
  pickle.dump(lr_study, f)

best_lr = lr_study.best_params['lr']
best_weight_decay = lr_study.best_params['weight_decay']
best_k = lr_study.best_params['k']
best_alpha = lr_study.best_params['alpha']
tft_objective = partial(
  opt_func.tft_objective,
  lr=best_lr,
  weight_decay=best_weight_decay,
  k=best_k,
  alpha=best_alpha,
  train_loader=train_loader,
  val_loader=val_loader
)

tft_study = optuna.create_study(direction='minimize')
tft_study.optimize(tft_objective, n_trials=20)

with open (STUDY_PATH + 'tft_study.pkl', 'wb') as f:
  pickle.dump(tft_study, f)