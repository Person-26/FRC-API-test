from torch import nn
import torch
import torch.nn.functional as F

class ScorePredictor(nn.Module):
    def __init__(self, num_players = 100000, num_years = 30, num_events = 1000, embedding_dim = 64, hidden_dim = 1024):
        super(ScorePredictor, self).__init__()
        self.teams = nn.Embedding(num_players, embedding_dim)
        self.year = nn.Embedding(num_years, embedding_dim)
        self.event = nn.Embedding(num_events, embedding_dim)
        self.teams.weight.data.uniform_(-1, 1)  # Initialize embeddings with random values
        self.year.weight.data.uniform_(-1, 1)  # Initialize year embeddings with random values
        self.event.weight.data.uniform_(-1, 1)  # Initialize event embeddings with random values
        self.fc1 = nn.Linear(embedding_dim * 8 + 1, hidden_dim)  # Input: 6 player embeddings, event embedding, year embedding, tournament level
        self.dropout1 = nn.Dropout(p=0.5) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)  # Reduce dimensions
        self.dropout2 = nn.Dropout(p=0.5) 
        self.fc3 = nn.Linear(hidden_dim // 2, 1)  # Second additional layer

        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, ids):
        # Convert player IDs to embeddings
        tournament_level = ids[:, 2:3]  # Extract the tournament level
        player_ids = ids[:, 3:9]  # Slice along the second dimension
        year = ids[:, 0:1]  # Extract the year
        event = ids[:, 1:2]
        year_embeddings = self.year(year)
        event_embeddings = self.event(event)
        player_embeddings = self.teams(player_ids)
        x = torch.cat((player_embeddings, event_embeddings, year_embeddings), dim=1)
        # Flatten player embeddings (combine all player embeddings into one vector per game)
        x = x.view(x.shape[0], -1)  # Shape: (batch_size, 6 * embedding_dim)
        x = torch.cat((x, tournament_level), dim=1)
        # Feed forward through layers with Leaky ReLU
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = self.dropout2(x)
        x = self.fc3(x)

        return x