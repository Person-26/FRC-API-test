# 6, 9990, 191, 1507, 1511, 1585, 1405

import torch
from ScorePredictor import ScorePredictor

model = ScorePredictor(10000, 16, 32, 6)
model.load_state_dict(torch.load("player_embeddings_model.pth"))
model.eval()

print(model(torch.tensor([6, 9990, 191, 1507, 1511, 1585, 1405])))