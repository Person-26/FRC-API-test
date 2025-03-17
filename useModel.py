import torch
from trainModel import NeuralNetwork

model = NeuralNetwork()
model.load_state_dict(torch.load('model.pth', weights_only=True))
test = torch.tensor([2024, hash('NYRoc'), 6, 9990, 191, 1507, 1511, 1585, 1405], dtype=torch.float32) 
prediction = model(test)
print(prediction)