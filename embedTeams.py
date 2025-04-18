import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from ScorePredictor import ScorePredictor
import keyboard  
import os
from test import test

def main():
    # Create model
    model = ScorePredictor()

    dataset = torch.load("dataset.pt", weights_only=False)  
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=os.cpu_count())

    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss with logits
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Adam optimizer with weight decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    model.to(device)
    criterion.to(device)
    # Training Loop
    epoch = 1
    while True:
        total_loss = 0
        for batch in dataloader:
            if keyboard.is_pressed('esc'):  # Check if 'Esc' key is pressed
                print("Training stopped by user.")
                break
            model.train()
            player_ids, target_scores = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            predictions = model(player_ids)
            loss = criterion(predictions, target_scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Convert raw logits to probabilities
            probabilities = torch.sigmoid(predictions)
            
        else:
            print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader):.4f}, "
                  f"Prediction: {probabilities[0].item():.4f}, Target: {target_scores[0].item():.4f}, Accuracy: {test(model, device):.4f}")
            epoch += 1
            scheduler.step()
            continue  # Continue to the next epoch if not interrupted
        break  # Break the outer loop if 'Esc' was pressed
    # Save model
    torch.save(model.state_dict(), "player_embeddings_model.pth")

if __name__ == '__main__':
    main()