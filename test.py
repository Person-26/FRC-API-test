
import torch
from ScorePredictor import ScorePredictor

# Create model
model = ScorePredictor()
model.load_state_dict(torch.load("player_embeddings_model.pth"))
model.eval()

result = torch.sigmoid(model(torch.tensor([[24, 804, 2067, 51, 8717, 3538, 118]])))

redScore = 79
blueScore = 114

print(result)
print(1 if (redScore > blueScore) else 0.5 if (redScore == blueScore) else 0)