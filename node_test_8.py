import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim

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

# Train on multiple mazes
def train_on_multiple_mazes(model, training_data, max_epochs=100, lr=0.01, patience=10):
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

        if total_loss < best_loss:
            best_loss = total_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} with best loss {best_loss:.4f}")
                break

    print(f"Final training loss: {total_loss:.4f}")

# Simulate path with exploration
def simulate_path_with_exploration(model, start, goal, node_to_idx, idx_to_node, graph, max_steps=100):
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
            print("No valid neighbors remain. Dead end reached.")
            break

        prob_values = list(neighbor_probs.values())
        prob_values = [p / sum(prob_values) for p in prob_values]
        next_node = list(neighbor_probs.keys())[prob_values.index(max(prob_values))]
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

# Adjust the model for the unseen maze
def adjust_model_for_unseen_maze(model, unseen_num_nodes, embedding_dim):
    current_embedding_nodes = model.node_embedding.num_embeddings
    if current_embedding_nodes != unseen_num_nodes:
        model.node_embedding = nn.Embedding(unseen_num_nodes, embedding_dim)
        nn.init.xavier_uniform_(model.node_embedding.weight)
        print(f"Adjusted model: Embedding layer updated to handle {unseen_num_nodes} nodes.")

    current_output_nodes = model.fc[-2].out_features
    if current_output_nodes != unseen_num_nodes:
        model.fc[-2] = nn.Linear(64, unseen_num_nodes)
        nn.init.xavier_uniform_(model.fc[-2].weight)
        model.fc[-2].bias.data.zero_()
        print(f"Adjusted model: Output layer updated to handle {unseen_num_nodes} nodes.")

    return model

# Main incremental training loop
def train_incrementally_with_multiple_mazes(num_mazes=100, maze_size=(5, 5), wall_prob=0.3, unseen_maze_size=(7, 7)):
    embedding_dim = 16
    model = None
    all_training_data = []

    for maze_idx in range(num_mazes):
        maze, start, goal = generate_random_maze(maze_size[0], maze_size[1], wall_prob)
        graph = maze_to_graph(maze)

        training_data = generate_training_data(graph, start, goal)
        node_to_idx = {node: i for i, node in enumerate(graph.nodes)}
        all_training_data.extend((node_to_idx[curr], node_to_idx[next_]) for curr, next_ in training_data)

        print(f"Maze {maze_idx + 1}: Start={start}, Goal={goal}")

        if model is None:
            num_nodes = len(graph.nodes)
            model = NodePredictor(num_nodes, embedding_dim)

    print("\nTraining on Combined Mazes:")
    train_on_multiple_mazes(model, all_training_data)

    print("\nTesting on Unseen Maze:")
    unseen_maze, unseen_start, unseen_goal = generate_random_maze(unseen_maze_size[0], unseen_maze_size[1], wall_prob)
    unseen_graph = maze_to_graph(unseen_maze)
    unseen_node_to_idx = {node: i for i, node in enumerate(unseen_graph.nodes)}
    unseen_idx_to_node = {i: node for node, i in unseen_node_to_idx.items()}

    unseen_num_nodes = len(unseen_graph.nodes)
    model = adjust_model_for_unseen_maze(model, unseen_num_nodes, embedding_dim)

    print("\nUnseen Maze:")
    visualize_solution(unseen_maze, [])

    print("\nSimulating Path on Unseen Maze:")
    unseen_path = simulate_path_with_exploration(model, unseen_start, unseen_goal, unseen_node_to_idx, unseen_idx_to_node, unseen_graph)
    visualize_solution(unseen_maze, unseen_path)

# Execute training
train_incrementally_with_multiple_mazes(num_mazes=, maze_size=(5, 5), wall_prob=0.3, unseen_maze_size=(7, 7))
