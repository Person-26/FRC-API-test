import torch
from ScorePredictor import ScorePredictor
import os
from torch.utils.data import DataLoader
import math

def main():
    # Load the pretrained model
    model = ScorePredictor()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()  # Set the model to evaluation mode

    data = torch.load("alldata.pt", weights_only=False)
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=os.cpu_count())
    teams = []

    for batch in dataloader:
        ids = batch[0]
        if ids[0][0].item() == 25 :  # Check if the event is 25 and level is 1
            for team in ids[0][3:9].tolist():
                if team not in teams:
                    teams.append(team)
    team = 6731
    # Extract embeddings
    print(min([(i, math.sqrt(torch.sum((model.teams(torch.tensor([team]))- model.teams(torch.tensor([i]))) ** 2, dim=-1).item()) if i != team else 999)for i in teams], key=lambda x: x[1]))

if __name__ == "__main__":
    main()