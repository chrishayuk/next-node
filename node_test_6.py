import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.model_selection import train_test_split
from maze.maze_generator import generate_random_maze, print_maze
from maze.maze_to_graph_converter import maze_to_graph

# Helper: Dynamic loss scaling
def dynamic_loss_scaling(loss, epoch, total_epochs):
    scaling_factor = max(0.5, 1 - (epoch / total_epochs))
    return loss * scaling_factor

# Step 1: Node Predictor Model
class NodePredictor(nn.Module):
    def __init__(self, num_nodes, embedding_dim=32):
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout for regularization
            nn.Linear(64, num_nodes),
            nn.Softmax(dim=1)
        )

    def forward(self, current_node_idx):
        embedding = self.node_embedding(current_node_idx)
        return self.fc(embedding)

# Step 2: Generate Paths and Training Data
def generate_paths(G, start, goal):
    return nx.shortest_path(G, source=start, target=goal)

def prepare_training_data(graph, start, goal):
    path = generate_paths(graph, start, goal)
    training_data = []
    for i in range(len(path) - 1):
        training_data.append((path[i], path[i + 1]))
    return training_data

def prepare_batch_training_data(mazes):
    training_data = []
    for maze, start, goal in mazes:
        graph = maze_to_graph(maze)
        node_to_idx = {node: i for i, node in enumerate(graph.nodes)}
        path = generate_paths(graph, start, goal)
        training_data.extend([(node_to_idx[curr], node_to_idx[next_]) for curr, next_ in zip(path[:-1], path[1:])])
    return training_data

# Step 3: Training Function
def train_model(model, training_data, validation_data, num_nodes, num_epochs=150, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for current_idx, next_idx in training_data:
            current_idx_tensor = torch.tensor([current_idx])
            next_idx_tensor = torch.tensor([next_idx])

            optimizer.zero_grad()
            output = model(current_idx_tensor)
            loss = criterion(output, next_idx_tensor)

            # Apply dynamic loss scaling
            loss = dynamic_loss_scaling(loss, epoch, num_epochs)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

        # Evaluate on validation data
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for current_idx, next_idx in validation_data:
                current_idx_tensor = torch.tensor([current_idx])
                next_idx_tensor = torch.tensor([next_idx])
                output = model(current_idx_tensor)
                validation_loss += criterion(output, next_idx_tensor).item()
        if epoch % 10 == 0:
            print(f"Validation Loss: {validation_loss:.4f}")

# Step 4: Prediction and Path Simulation
def predict_next_node(model, current_node, node_to_idx, idx_to_node, graph, visited):
    current_idx = torch.tensor([node_to_idx[current_node]])
    next_probs = model(current_idx).detach().numpy().squeeze()

    neighbors = list(graph.neighbors(current_node))
    neighbor_probs = {neighbor: next_probs[node_to_idx[neighbor]] for neighbor in neighbors if neighbor not in visited}

    if not neighbor_probs:
        return None  # Dead end
    return max(neighbor_probs, key=neighbor_probs.get)  # Node with the highest probability

def simulate_path_with_fallback(model, start, goal, node_to_idx, idx_to_node, graph, max_steps=100):
    current_node = start
    path = [current_node]
    visited = set()

    for _ in range(max_steps):
        visited.add(current_node)
        next_node = predict_next_node(model, current_node, node_to_idx, idx_to_node, graph, visited)

        if next_node is None:  # Dead end: attempt a fallback
            neighbors = [neighbor for neighbor in graph.neighbors(current_node) if neighbor not in visited]
            next_node = random.choice(neighbors) if neighbors else None

        if next_node is None:  # No valid moves even after fallback
            print("No valid moves. Simulation stopped.")
            break

        path.append(next_node)
        current_node = next_node
        if current_node == goal:
            print("Goal reached!")
            break

    return path

# Step 5: Visualization
def visualize_solution(maze, solution_path):
    maze_copy = [row[:] for row in maze]
    for r, c in solution_path:
        if maze_copy[r][c] not in ['S', 'G']:
            maze_copy[r][c] = '*'
    for row in maze_copy:
        print(' '.join(row))

# Main execution
# Generate the training and validation mazes
train_mazes = [(generate_random_maze(7, 7, wall_prob=0.3)) for _ in range(80)]
val_mazes = [(generate_random_maze(7, 7, wall_prob=0.3)) for _ in range(20)]

# Prepare training and validation data
train_data = prepare_batch_training_data(train_mazes)
val_data = prepare_batch_training_data(val_mazes)

# Initialize and train the model
num_nodes = max(len(maze_to_graph(maze[0]).nodes) for maze in train_mazes)
embedding_dim = 16
model = NodePredictor(num_nodes, embedding_dim)
train_model(model, train_data, val_data, num_nodes)

# Test on an unseen maze
unseen_maze, unseen_start, unseen_goal = generate_random_maze(7, 7, wall_prob=0.3)
unseen_graph = maze_to_graph(unseen_maze)
unseen_node_to_idx = {node: i for i, node in enumerate(unseen_graph.nodes)}
unseen_idx_to_node = {i: node for node, i in unseen_node_to_idx.items()}

print("\nUnseen Maze:")
print_maze(unseen_maze)

print("\nSimulating Path on Unseen Maze:")
unseen_path = simulate_path_with_fallback(model, unseen_start, unseen_goal, unseen_node_to_idx, unseen_idx_to_node, unseen_graph)
visualize_solution(unseen_maze, unseen_path)
