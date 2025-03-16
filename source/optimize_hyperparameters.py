from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import pickle
from settings import *
import data

_, train_loader, val_loader, _ = data.get_time_series()

study = optimize_hyperparameters(
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
    model_path=MODEL_PATH,
    n_trials=100,
    max_epochs=20,
    gradient_clip_val_range=(0.01, 1.0),
    hidden_size_range=(8, 128),
    hidden_continuous_size_range=(8, 128),
    attention_head_size_range=(1, 4),
    learning_rate_range=(1e-4, 1e-2),
    dropout_range=(0.1, 0.3),
    use_learning_rate_finder=True,
)
with open (STUDY_PATH + 'hp_study.pkl', 'wb') as f:
  pickle.dump(study, f)