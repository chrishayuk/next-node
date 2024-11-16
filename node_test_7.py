import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import random

from maze.maze_to_graph_converter import maze_to_graph
from maze.maze_generator import generate_random_maze

# Node Predictor Model
class NodePredictor(nn.Module):
    def __init__(self, num_nodes, embedding_dim=16):
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

# Generate paths and training data
def generate_paths(G, start, goal):
    return nx.shortest_path(G, source=start, target=goal)

def generate_training_data(G, start, goal):
    path = generate_paths(G, start, goal)
    training_data = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    return training_data

# Train on a single maze
def train_on_maze(model, training_data, num_nodes, max_epochs=100, lr=0.01, patience=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        total_loss = 0
        for current_node, next_node in training_data:
            current_node_tensor = torch.tensor([current_node])
            next_node_tensor = torch.tensor([next_node])

            optimizer.zero_grad()
            output = model(current_node_tensor)
            loss = criterion(output, next_node_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Early stopping condition
        if total_loss < best_loss:
            best_loss = total_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_loss

# Simulate path on a maze
def simulate_path(model, start, goal, node_to_idx, idx_to_node, graph, max_steps=100):
    current_node = start
    path = [current_node]
    visited = set()

    for _ in range(max_steps):
        visited.add(current_node)
        current_idx = torch.tensor([node_to_idx[current_node]])
        next_probs = model(current_idx).detach().numpy().squeeze()

        neighbors = list(graph.neighbors(current_node))
        neighbor_probs = {neighbor: next_probs[node_to_idx[neighbor]] for neighbor in neighbors if neighbor not in visited}

        if not neighbor_probs:
            print("Dead end encountered. Path simulation stopped.")
            break

        next_node = max(neighbor_probs, key=neighbor_probs.get)
        path.append(next_node)
        current_node = next_node

        if current_node == goal:
            print("Goal reached!")
            break

    return path

# Visualize the solution
def visualize_solution(maze, solution_path):
    maze_copy = [row[:] for row in maze]
    for r, c in solution_path:
        if maze_copy[r][c] not in ['S', 'G']:
            maze_copy[r][c] = '*'
    for row in maze_copy:
        print(' '.join(row))

# Incremental Training Loop
def train_incrementally(num_mazes=5, maze_size=(5, 5), wall_prob=0.3):
    embedding_dim = 16
    for maze_idx in range(num_mazes):
        # Generate maze
        maze, start, goal = generate_random_maze(maze_size[0], maze_size[1], wall_prob)
        graph = maze_to_graph(maze)

        # Prepare data
        training_data = generate_training_data(graph, start, goal)
        node_to_idx = {node: i for i, node in enumerate(graph.nodes)}
        idx_to_node = {i: node for node, i in node_to_idx.items()}
        training_indices = [(node_to_idx[curr], node_to_idx[next_]) for curr, next_ in training_data]
        num_nodes = len(graph.nodes)

        # Initialize or reuse model
        model = NodePredictor(num_nodes, embedding_dim)

        print(f"\nTraining on Maze {maze_idx + 1}:")
        train_on_maze(model, training_indices, num_nodes)

        print("\nSimulating Path on Maze:")
        simulated_path = simulate_path(model, start, goal, node_to_idx, idx_to_node, graph)
        visualize_solution(maze, simulated_path)

# Execute incremental training
train_incrementally(num_mazes=3, maze_size=(7, 7), wall_prob=0.3)
