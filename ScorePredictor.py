from torch import nn
import torch
import torch.nn.functional as F

class ScorePredictor(nn.Module):
    def __init__(self, num_players = 10000, num_years = 30, embedding_dim = 64, hidden_dim = 2048):
        super(ScorePredictor, self).__init__()
        self.teams = nn.Embedding(num_players, embedding_dim)
        self.year = nn.Embedding(num_years, embedding_dim)
        self.teams.weight.data.uniform_(-1, 1)  # Initialize embeddings with random values
        self.year.weight.data.uniform_(-1, 1)  # Initialize year embeddings with random values
        self.fc1 = nn.Linear(embedding_dim * 7, hidden_dim)  # Input: 6 player embeddings
        self.fc2 = nn.Linear(hidden_dim, 1)  # Second hidden layer

        for layer in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, ids):
        # Convert player IDs to embeddings
        player_ids = ids[:, 1:8]  # Slice along the second dimension
        year = ids[:, 0:1]  # Extract the year
        year_embeddings = self.year(year)
        player_embeddings = self.teams(player_ids.long())
        x = torch.cat((player_embeddings, year_embeddings), dim=1)
        # Flatten player embeddings (combine all player embeddings into one vector per game)
        x = x.view(x.shape[0], -1)  # Shape: (batch_size, 6 * embedding_dim)
        # Feed forward through layers with Leaky ReLU
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.fc2(x)

        return x