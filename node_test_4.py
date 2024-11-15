import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import json
from maze.maze_dataset_generator import MazeDatasetGenerator
from maze.maze_to_graph_converter import maze_to_graph

# Initialize dataset generator parameters
num_mazes = 100
min_size = 5
max_size = 7
output_path = "datasets/maze/training/maze_training_data.jsonl"

# Generate maze dataset
dataset_generator = MazeDatasetGenerator(num_mazes, min_size, max_size, output_path)
dataset_generator.generate_maze_dataset()

# Load the generated dataset
def load_maze_dataset(dataset_path):
    with open(dataset_path, "r") as f:
        return [json.loads(line) for line in f]

# Prepare data for training
def prepare_training_data(dataset):
    training_data = []
    node_mappings = []
    max_node_index = 0  # To track the highest node index

    for entry in dataset:
        graph = maze_to_graph(entry["context_representation"])  # Recreate the graph from the maze
        node_to_idx = {node: i for i, node in enumerate(graph.nodes)}
        idx_to_node = {i: node for node, i in node_to_idx.items()}

        # Update max node index for all graphs
        max_node_index = max(max_node_index, max(node_to_idx.values()))

        path = [eval(node) for node in entry["path"]]  # Convert string nodes back to tuples
        training_indices = [(node_to_idx[curr], node_to_idx[next_]) for curr, next_ in zip(path[:-1], path[1:])]
        training_data.extend(training_indices)
        node_mappings.append((node_to_idx, idx_to_node, graph))

    return training_data, max_node_index + 1, node_mappings  # Ensure num_nodes includes all nodes

# Training function using the dataset
def train_model_on_dataset(model, training_data, num_nodes, num_epochs=100, lr=0.001):
    embedding_dim = 16
    model = NodePredictor(num_nodes, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for current_idx, next_idx in training_data:
            current_idx_tensor = torch.tensor([current_idx])
            next_idx_tensor = torch.tensor([next_idx])

            # Ensure target index is within bounds
            assert next_idx_tensor.item() < num_nodes, f"Target {next_idx_tensor.item()} is out of bounds for {num_nodes} nodes."

            optimizer.zero_grad()
            output = model(current_idx_tensor)
            loss = criterion(output, next_idx_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    return model

# Prediction and path simulation
def predict_next_node(model, current_node, node_to_idx, idx_to_node, graph, visited):
    current_idx = torch.tensor([node_to_idx[current_node]])
    next_probs = model(current_idx).detach().numpy().squeeze()

    neighbors = list(graph.neighbors(current_node))
    neighbor_probs = {neighbor: next_probs[node_to_idx[neighbor]] for neighbor in neighbors if neighbor not in visited}

    if not neighbor_probs:
        return None  # Dead end
    return max(neighbor_probs, key=neighbor_probs.get)  # Node with the highest probability

def simulate_path(model, start, goal, node_to_idx, idx_to_node, graph, max_steps=100):
    current_node = start
    path = [current_node]
    visited = set()

    for _ in range(max_steps):
        visited.add(current_node)
        next_node = predict_next_node(model, current_node, node_to_idx, idx_to_node, graph, visited)
        if next_node is None:
            print("Dead end encountered. Path simulation stopped.")
            break
        path.append(next_node)
        current_node = next_node
        if current_node == goal:
            print("Goal reached!")
            break
    return path

# Visualization function
def visualize_solution(maze, solution_path):
    maze_copy = [row[:] for row in maze]
    for r, c in solution_path:
        if maze_copy[r][c] not in ['S', 'G']:
            maze_copy[r][c] = '*'
    for row in maze_copy:
        print(' '.join(row))

# Step 3: Next-Node Prediction Model
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

# Main execution
dataset_path = output_path
dataset = load_maze_dataset(dataset_path)
training_data, num_nodes, node_mappings = prepare_training_data(dataset)

# Log dataset details
print(f"Number of mazes: {len(dataset)}")
print(f"Total number of nodes: {num_nodes}")
print(f"Total training pairs: {len(training_data)}")

# Train the model using the dataset
model = train_model_on_dataset(None, training_data, num_nodes)

# Simulate path on a sample maze from the dataset
sample_maze_data = dataset[0]
sample_maze = sample_maze_data["context_representation"]
sample_start = eval(sample_maze_data["path"][0])
sample_goal = eval(sample_maze_data["path"][-1])

# Retrieve the corresponding graph and mappings
node_to_idx, idx_to_node, graph = node_mappings[0]

print("\nSimulating Path on Sample Maze:")
simulated_path = simulate_path(model, sample_start, sample_goal, node_to_idx, idx_to_node, graph)
visualize_solution(sample_maze, simulated_path)
