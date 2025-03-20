import json
import torch

# List to store structured data
games = []
scores = []

# Load json
for year in range(2009, 2025):
    with open("matches/"+ str(year)+".json", "r") as f:
        data = json.load(f)

    # Extract matches
    for event in data.values():  # Loop through all events
        for match in event:  # Loop through matches in each event
            teams = match["teams"]  # Extract teams

            # Extract player IDs
            players = [player["teamNumber"] for player in teams]

            # Extract score (assuming stored in match["score"])
            score = [match["scoreRedFinal"], match["scoreRedFoul"], match["scoreRedAuto"], match["scoreBlueFinal"], match["scoreBlueFoul"], match["scoreBlueAuto"]]
            if not score.__contains__(None):
                games.append(players)
                scores.append(score)



# Convert to PyTorch tensors
games_tensor = torch.tensor(games, dtype=torch.int)  # Player IDs
scores_tensor = torch.tensor(scores, dtype=torch.float)  # Score vectors

