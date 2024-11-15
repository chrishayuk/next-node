import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim

from maze.maze_to_graph_converter import maze_to_graph
from maze.maze_generator import generate_random_maze

# Generate a training maze
maze, start, goal = generate_random_maze(5, 5, wall_prob=0.3)

# Generate a testing maze (with different parameters if desired)
new_maze, new_start, new_goal = generate_random_maze(6, 6, wall_prob=0.4)


# Step 2: Generate Paths and Training Data
def generate_paths(G, start, goal):
    return nx.shortest_path(G, source=start, target=goal)

def generate_training_data(G, start, goal):
    path = generate_paths(G, start, goal)
    training_data = []
    for i in range(len(path) - 1):
        training_data.append((path[i], path[i + 1]))
    return training_data

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

# Step 4: Training Function
def train_model(model, training_indices, num_epochs=100, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0
        for current_idx, next_idx in training_indices:
            current_idx_tensor = torch.tensor([current_idx])
            next_idx_tensor = torch.tensor([next_idx])

            optimizer.zero_grad()
            output = model(current_idx_tensor)
            loss = criterion(output, next_idx_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# Step 5: Path Simulation
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

# Step 6: Visualization
def visualize_solution(maze, solution_path):
    maze_copy = [row[:] for row in maze]
    for r, c in solution_path:
        if maze_copy[r][c] not in ['S', 'G']:
            maze_copy[r][c] = '*'
    for row in maze_copy:
        print(' '.join(row))

# Step 7: Train and Test on Original Maze
print("\nTraining on Original Maze:")
graph = maze_to_graph(maze)
start = next((r, c) for r, row in enumerate(maze) for c, cell in enumerate(row) if cell == 'S')
goal = next((r, c) for r, row in enumerate(maze) for c, cell in enumerate(row) if cell == 'G')

training_data = generate_training_data(graph, start, goal)
node_to_idx = {node: i for i, node in enumerate(graph.nodes)}
idx_to_node = {i: node for node, i in node_to_idx.items()}
training_indices = [(node_to_idx[curr], node_to_idx[next_]) for curr, next_ in training_data]

num_nodes = len(graph.nodes)
embedding_dim = 16
model = NodePredictor(num_nodes, embedding_dim)

train_model(model, training_indices)

print("\nSimulating Path on Original Maze:")
simulated_path = simulate_path(model, start, goal, node_to_idx, idx_to_node, graph)
visualize_solution(maze, simulated_path)

# Step 8: Test on Unseen Maze
print("\nTesting on Unseen Maze:")
new_graph = maze_to_graph(new_maze)
new_start = next((r, c) for r, row in enumerate(new_maze) for c, cell in enumerate(row) if cell == 'S')
new_goal = next((r, c) for r, row in enumerate(new_maze) for c, cell in enumerate(row) if cell == 'G')

new_training_data = generate_training_data(new_graph, new_start, new_goal)
new_node_to_idx = {node: i for i, node in enumerate(new_graph.nodes)}
new_idx_to_node = {i: node for node, i in new_node_to_idx.items()}
new_training_indices = [(new_node_to_idx[curr], new_node_to_idx[next_]) for curr, next_ in new_training_data]

new_num_nodes = len(new_graph.nodes)
new_model = NodePredictor(new_num_nodes, embedding_dim)

train_model(new_model, new_training_indices)

print("\nSimulating Path on Unseen Maze:")
new_simulated_path = simulate_path(new_model, new_start, new_goal, new_node_to_idx, new_idx_to_node, new_graph)
visualize_solution(new_maze, new_simulated_path)
