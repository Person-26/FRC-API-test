import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from teamToMatch import games_tensor, scores_tensor
from ScorePredictor import ScorePredictor

# Create model
model = ScorePredictor()

dataset = TensorDataset(games_tensor, scores_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss with logits
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")
model.to(device)

# Training Loop
epoch = 0
limit = 10
while epoch < limit:
    total_loss = 0
    for batch in dataloader:
        player_ids, target_scores = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        predictions = model(player_ids)
        loss = criterion(predictions, target_scores)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Convert raw logits to probabilities
        probabilities = torch.sigmoid(predictions)
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}, "
              f"Prediction: {probabilities[0].item():.4f}, Target: {target_scores[0].item():.4f}")
        
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