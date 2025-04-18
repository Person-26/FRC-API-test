import torch
from torch.utils.data import DataLoader
from ScorePredictor import ScorePredictor
import os
import torch.nn as nn

class Correct(nn.Module):
    def __init__(self):
        super(Correct, self).__init__()

    def forward(self, predictions, targets):
        return (torch.eq(torch.round(predictions), targets)).float()
    
def test (model, device):
    model.eval()
    total_accuracy = 0
    total_batches = 0
    testSet = torch.load("testSet.pt", weights_only=False)
    dataloader = DataLoader(testSet, batch_size=64, shuffle=True, num_workers=os.cpu_count())
    criterion = Correct()
    criterion.to(device)

    with torch.no_grad():
        for batch in dataloader:
            ids, scores = batch[0].to(device), batch[1].to(device)
            predictions = torch.sigmoid(model(ids))
            accuracy = criterion(predictions, scores)
            total_accuracy += accuracy.mean().item()
            total_batches += 1

    overall_accuracy = total_accuracy / total_batches if total_batches > 0 else 0
    return overall_accuracy

def main ():
    # Create model
    model = ScorePredictor()
    model.load_state_dict(torch.load("player_embeddings_model.pth"))
    model.eval()

    testSet = torch.load("testSet.pt", weights_only=False)
    dataloader = DataLoader(testSet, batch_size=64, shuffle=True, num_workers=os.cpu_count())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    criterion = Correct()
    criterion.to(device)

    total_accuracy = 0
    total_batches = 0

    for batch in dataloader:
        ids, scores = batch[0].to(device), batch[1].to(device)
        predictions = torch.sigmoid(model(ids))
        accuracy = criterion(predictions, scores)
        total_accuracy += accuracy.mean().item()
        total_batches += 1

    overall_accuracy = total_accuracy / total_batches if total_batches > 0 else 0
    print(f"Overall Accuracy: {overall_accuracy}")

if __name__ == "__main__":
    main()