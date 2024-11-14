import torch.nn as nn

class NodePredictor(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_nodes),
            nn.Softmax(dim=1)
        )

    def forward(self, current_node_idx):
        embedding = self.node_embedding(current_node_idx)
        return self.fc(embedding)
