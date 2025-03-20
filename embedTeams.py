import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import json
from teamToMatch import games_tensor, scores_tensor
from ScorePredictor import ScorePredictor

# Load the player list from 2024teams.json
with open("2024teams.json", "r") as f:
    player_list = json.load(f)

# Define constants
num_players = max(player_list) + 1  # Total unique players
embedding_dim = 16  # Size of player embeddings
hidden_dim = 32  # Hidden layer size
output_dim = 6  # Predicting 6 types of points

# Create model
model = ScorePredictor(num_players, embedding_dim, hidden_dim, output_dim)

dataset = TensorDataset(games_tensor, scores_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epoch = 0
limit = 100
while epoch < limit:
    total_loss = 0
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in dataloader:
        player_ids, target_scores = batch
        optimizer.zero_grad()
        predictions = model(player_ids)
        loss = criterion(predictions, target_scores)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")
        
        if epoch == limit:
            # Prompt user to continue after each epoch
            response = input(f"Epoch {epoch+1} completed. How many more epochs would you like to train? Enter a number or 'stop' to end training: ")
            if response.lower() == 'stop':
                print("Training stopped by user.")
                break
            else:
                limit += int(response)
        epoch += 1

# Save model
torch.save(model.state_dict(), "player_embeddings_model.pth")