import torch
from torch.utils.data import DataLoader
from ScorePredictor import ScorePredictor
from test import Correct
import os
import torch.nn as nn
import json

with open("events.json", "r") as f:
    events = json.load(f)

def main ():
    # Create model
    model = ScorePredictor()
    model.load_state_dict(torch.load("model.pth"))

    device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    total_accuracy = 0
    total_batches = 0
    test_set = torch.load("testSet.pt", weights_only=False)
    data_set = torch.load("dataset.pt", weights_only=False)
    dataloader1 = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=os.cpu_count())
    dataloader2 = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=os.cpu_count())
    criterion = Correct()
    criterion.to(device)
    for batch in dataloader1:
        ids, scores = batch[0].to(device), batch[1].to(device)
        if ids[0][0].item() == 25 and 6731 in ids[0][3:9]:  
            predictions = torch.sigmoid(model(ids))
            accuracy = criterion(predictions, scores)
            print(f"Event: {events[ids[0][1]]}, Level: {"Qual" if(ids[0][2] == 0) else "Play"}, Team Numbers: {[id.item() for id in ids[0][3:9]]}, Prediction: {predictions[0].item()}, Actual: {scores[0].item()}, Correct: {bool(accuracy[0].item())}")
            total_accuracy += accuracy.mean().item()
            total_batches += 1
    for batch in dataloader2:
        ids, scores = batch[0].to(device), batch[1].to(device)
        if ids[0][0].item() == 25 and 6731 in ids[0][3:9]:  
            predictions = torch.sigmoid(model(ids))
            accuracy = criterion(predictions, scores)
            print(f"Event: {events[ids[0][1]]}, Level: {"Qual" if(ids[0][2] == 0) else "Play"}, Team Numbers: {[id.item() for id in ids[0][3:9]]}, Prediction: {predictions[0].item()}, Actual: {scores[0].item()}, Correct: {bool(accuracy[0].item())}")
            total_accuracy += accuracy.mean().item()
            total_batches += 1

    overall_accuracy = total_accuracy / total_batches if total_batches > 0 else 0
    print(overall_accuracy)

if __name__ == "__main__":
    main()