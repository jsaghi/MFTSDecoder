from settings import *
import data
from expanders import Expander36X, Expander6X


# Function to train and evaluate 
def expander_loop(ratio, num_epochs):

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if ratio == 6:
    expander = Expander6X((1, IF_LENGTH))
  else:
    expander = Expander36X((1, LF_LENGTH))
  expander.to(device)

  optimizer = torch.optim.Adam(expander.parameters(), lr=1e-4)
  loss_fn = nn.MSELoss()

  train, val, test = data.get_temp(ratio)

  for epoch in range(num_epochs):
    expander.train(True)
    running_loss = 0.0

    for i, (inputs, targets) in enumerate(train):
      inputs, targets = inputs.to(device), targets.to(device)
      optimizer.zero_grad()
      outputs = expander(inputs)
      loss = loss_fn(outputs, targets)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

      if i % 1000 == 999:
        print(f'Epoch {epoch + 1}, Step [{i + 1}/{len(train)}], Loss: {loss.item():.4f}')

    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train)}')

    expander.train(False)
    val_loss = 0.0
    with torch.no_grad():
      for inputs, targets in val:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = expander(inputs)
        loss = loss_fn(outputs, targets)
        val_loss += loss.item()

    print(f'Val Loss: {val_loss / len(val)}')
