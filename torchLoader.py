import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from random import random

# Define a simple neural network
class TrajectoryModel(nn.Module):
    def __init__(self):
        super(TrajectoryModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 64),  # Input: time t
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)   # Output: [x, y]
        )

    def forward(self, t):
        return self.fc(t)

# Training the model
if __name__ == "__main__":
    # Create dataset and dataloader
    dataset = ParabolicTrajectoryDataset(num_points=100)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = TrajectoryModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop with convergence condition
    max_epochs = 1000  # Set a maximum number of epochs to avoid infinite loops
    loss_threshold = 1e-4  # Stop training when loss is below this value
    epoch = 0

    while epoch < max_epochs:
        total_loss = 0
        for t, target in dataloader:
            t = t.unsqueeze(1)  # Add a dimension for the input (batch_size, 1)
            target = target  # Target is [x, y]

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

        # Check for convergence
        if avg_loss < loss_threshold:
            print("Training converged!")
            break

        epoch += 1

    # Save the trained model
    model_path = "model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Test the model
    test_time = torch.tensor([[0.5]], dtype=torch.float32)  # Example time input
    predicted_position = model(test_time)
    print(f"Predicted position at t=0.5: {predicted_position}")

