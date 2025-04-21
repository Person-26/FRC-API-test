import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from ScorePredictor import ScorePredictor
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
    best_accuracy = 0
    decrease_count = 0  # Counter for consecutive accuracy decreases

    while True:
        total_loss = 0
        for batch in dataloader:
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
        
        # Code previously in the else block
        accuracy = test(model, device)
        
        # Check if accuracy decreased
        if accuracy < best_accuracy:
            decrease_count += 1
        else:
            decrease_count = 0
            best_accuracy = accuracy
            torch.save(model.state_dict(), "model.pth")  # Save the best model immediately
            
        print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader):.4f}, "
              f"Accuracy: {accuracy:.4f}, Best Accuracy: {best_accuracy:.4f}")
        
        if decrease_count >= 2:
            print("Stopping training due to accuracy decrease")
            break

        epoch += 1
        scheduler.step()
        continue  # Continue to the next epoch if not interrupted

if __name__ == '__main__':
    main()