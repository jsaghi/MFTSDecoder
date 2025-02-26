from settings import *
import data
import torch
import torch.nn as nn
import json


# Function to train and evaluate expander models 
def expander_loop(model, ratio, num_epochs, save_name):

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  loss_fn = nn.MSELoss()

  train, val, _ = data.get_temp(ratio)

  best_val_loss = float('inf')
  history = {'train_loss': [], 'val_loss': []}

  for epoch in range(num_epochs):
    model.train(True)
    running_loss = 0.0

    for i, (inputs, targets) in enumerate(train):
      inputs, targets = inputs.to(device), targets.to(device)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = loss_fn(outputs, targets)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

      if i % 1000 == 999:
        print(f'Epoch {epoch + 1}, Step [{i + 1}/{len(train)}], Loss: {loss.item():.4f}')

    running_loss /= len(train)
    print(f'Epoch {epoch + 1}, Loss: {running_loss}')

    model.train(False)
    val_loss = 0.0
    with torch.no_grad():
      for inputs, targets in val:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        val_loss += loss.item()

    val_loss /= len(val)
    print(f'Val Loss: {val_loss}')

    history['train_loss'].append(running_loss)
    history['val_loss'].append(val_loss)

    history_save_path = f'{HISTORY_PATH + save_name}_{epoch}'
    with open(history_save_path, 'w') as f:
      json.dump(history, f)

    model_save_path = f'{MODEL_PATH + save_name}_{epoch}'
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      torch.save(model.state_dict(), model_save_path)
