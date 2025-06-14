import torch
import torch.nn as nn
from nesterov import Nesterov
from dowg import DoWG, CDoWG, NDoWG
import matplotlib.pyplot as plt


# 1. Example Data
torch.manual_seed(0)  # For reproducibility
X = torch.randn(100, 100)
y = torch.randn(100, 1)

# 2. Model Factory
def get_model():
    torch.manual_seed(0) # for model weight initialization
    return nn.Linear(100, 1)

# 3. Loss function
criterion = nn.MSELoss()

# 4. Training function
def train_model(optimizer_name, optimizer_class, model, **kwargs):
    print(f"Training with {optimizer_name} optimizer...")
    optimizer = optimizer_class(model.parameters(), **kwargs)
    losses = []
    num_epochs = 1000
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        losses.append(loss.item())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    print(f"Finished training with {optimizer_name}.\n")
    return losses

# 5. Run experiments and collect losses
optimizers_to_compare = {
    "SGD (Nesterov)": (torch.optim.SGD, {'lr': 0.001, 'momentum': 0.9, 'nesterov': True}),
    "DoWG": (DoWG, {}),
    "NDoWG": (NDoWG, {'eps': 1e-4, 'momentum': 0.9})
}

all_losses = {}
for name, (optimizer_class, kwargs) in optimizers_to_compare.items():
    model = get_model()
    losses = train_model(name, optimizer_class, model, **kwargs)
    all_losses[name] = losses

# 6. Plotting the results
plt.figure(figsize=(12, 8))
for name, losses in all_losses.items():
    plt.plot(losses, label=name)

plt.title('Optimizer Comparison: Loss vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('optimizer_comparison.png')
print("\nPlot saved as optimizer_comparison.png")

# To run this example, execute `python main.py` in your terminal.
