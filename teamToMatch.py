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
            if match["scoreRedFinal"] is not None and match["scoreBlueFinal"] is not None:  # Check if scores are available
                teams = match["teams"]  # Extract teams

                # Extract player IDs
                players = [year - 2000] +[player["teamNumber"] for player in teams]
                #players = [player["teamNumber"] for player in teams]

                # Extract score (assuming stored in match["score"])
                score = [1 if (match["scoreRedFinal"] > match["scoreBlueFinal"]) else 0.5 if (match["scoreRedFinal"] == match["scoreBlueFinal"]) else 0]
                games.append(players)
                scores.append(score)



# Convert to PyTorch tensors
games_tensor = torch.tensor(games, dtype=torch.int)  # Player IDs
scores_tensor = torch.tensor(scores, dtype=torch.float)  # Score vectors