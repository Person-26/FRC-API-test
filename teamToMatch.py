import json
import torch
from torch.utils.data import TensorDataset
import random

# List to store structured data
games = []
scores = []
events = []
test_games = []
test_scores = []

# Load json
for year in range(2015, 2026):
    with open("matches/"+ str(year)+".json", "r") as f:
        data = json.load(f)

    # Extract matches
    for eventName, event in data.items():  # Loop through all events
        for match in event:  # Loop through matches in each event
            if match["scoreRedFinal"] is not None and match["scoreBlueFinal"] is not None:  # Check if scores are available
                teams = match["teams"]  # Extract teams

                if(eventName not in events):
                    events.append(eventName)

                # Extract player IDs
                players = [year - 2000, events.index(eventName), 0 if match["tournamentLevel"] == "Qualification" else 1] +[player["teamNumber"] for player in teams]
                #players = [player["teamNumber"] for player in teams]

                # Extract score (assuming stored in match["score"])
                score = [1 if (match["scoreRedFinal"] > match["scoreBlueFinal"]) else 0.5 if (match["scoreRedFinal"] == match["scoreBlueFinal"]) else 0]
                if(random.random() < 0.05):
                    test_games.append(players)
                    test_scores.append(score)
                else:
                    games.append(players)
                    scores.append(score)

with open("events.json", "w") as f:
    json.dump(events, f)

dataset = TensorDataset(torch.tensor(games, dtype=torch.int), torch.tensor(scores, dtype=torch.float))
torch.save(dataset, "dataset.pt")
testset = TensorDataset(torch.tensor(test_games, dtype=torch.int), torch.tensor(test_scores, dtype=torch.float))
torch.save(testset, "testset.pt")
alldataset = TensorDataset(torch.tensor(games + test_games, dtype=torch.int), torch.tensor(scores + test_scores, dtype=torch.float))
torch.save(alldataset, "alldata.pt")