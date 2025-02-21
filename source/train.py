from settings import *
from expanders import Expader36X, Expander6X

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decoder = Expander6X((1, 168))
decoder.to(device)

optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

for epoch in range(10):
  decoder.train(True)
  running_loss = 0.0

  for i, (inputs, targets) in enumerate(train_loader):
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = decoder(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

    if i % 1000 == 999:
      print(f'Epoch {epoch + 1}, Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')


  print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

  #torch.save(decoder.state_dict(), save_path + f'train_torch_decoder{epoch + 1}.pth')

  decoder.train(False)
  val_loss = 0.0
  with torch.no_grad():
    for inputs, targets in val_loader:
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = decoder(inputs)
      loss = loss_fn(outputs, targets)
      val_loss += loss.item()

  print(f'Val Loss: {val_loss / len(val_loader)}')