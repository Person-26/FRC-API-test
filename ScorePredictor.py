from torch import nn
import torch

class ScorePredictor(nn.Module):
    def __init__(self, num_players, embedding_dim, hidden_dim, output_dim):
        super(ScorePredictor, self).__init__()
        self.embedding = nn.Embedding(num_players, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 6, hidden_dim)  # Input: 6 player embeddings
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Output: 6-dimensional score vector

    def forward(self, player_ids):
        """
        player_ids: Tensor of shape (batch_size, 6) containing player IDs
        """
        # Convert player IDs to embeddings
        player_embeddings = self.embedding(player_ids)  # Shape: (batch_size, 6, embedding_dim)

        # Flatten player embeddings (combine all player embeddings into one vector per game)
        x = player_embeddings.view(player_embeddings.size(0), -1)  # Shape: (batch_size, 6 * embedding_dim)

        # Feed forward through layers
        x = torch.relu(self.fc1(x))
        scores = self.fc2(x)  # Output: 6-element score vector
        return scores