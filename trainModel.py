import torch
from torch.utils.data import DataLoader
from torch import nn, optim

# Define a simple neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(9, 512),  # Change input size from 28*28 to 9
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 6),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

# Training the model
if __name__ == "__main__":
    # Create dataset and dataloader
    # Load dataset from dataSet.pt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    dataset = torch.load("dataSet.pt", weights_only=False)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = NeuralNetwork().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop with convergence condition
    max_epochs = 1000  # Set a maximum number of epochs to avoid infinite loops
    loss_threshold = 1e-4  # Stop training when loss is below this value
    epoch = 0

    while epoch < max_epochs:
        total_loss = 0
        for t, target in dataloader:
            # Move tensors to the same device as the model
            t = t.to(device, dtype=torch.float32)  # Ensure input tensor is on the correct device
            target = target.to(device, dtype=torch.float32)  # Ensure target tensor is on the correct device

            # Forward pass
            output = model(t)
            loss = criterion(output, target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.6f}")

        epoch += 1

    # Save the trained model
    model_path = "model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

