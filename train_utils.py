import torch
import random
from bisect import insort


def train_model(model, training_indices, num_epochs=100, lr=0.01):
    """
    Train the NodePredictor model using CrossEntropyLoss.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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


def manhattan_distance(node1, node2):
    """
    Compute Manhattan distance between two nodes.
    """
    return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])


def predict_next_node_with_bias(model, current_node, node_to_idx, idx_to_node, graph, visited, goal):
    """
    Predict the next node with goal bias using Manhattan distance as a heuristic.
    """
    current_idx = torch.tensor([node_to_idx[current_node]])
    next_probs = model(current_idx).detach().numpy().squeeze()

    neighbors = list(graph.neighbors(current_node))
    neighbor_scores = {}
    for neighbor in neighbors:
        if neighbor not in visited:
            distance_to_goal = manhattan_distance(neighbor, goal)
            neighbor_scores[neighbor] = next_probs[node_to_idx[neighbor]] / (distance_to_goal + 1e-6)

    if not neighbor_scores:
        return None  # Dead end
    return max(neighbor_scores, key=neighbor_scores.get)


def simulate_path_with_pruning_and_exploration(
    model, start, goal, node_to_idx, idx_to_node, graph, explore=True, max_steps=200
):
    """
    Simulate a path with backtracking, pruning, goal bias, and optional exploration.
    """
    current_node = start
    path = [current_node]
    visited = set()
    stack = []  # For prioritized backtracking

    for step in range(max_steps):
        visited.add(current_node)

        # Predict the next node
        next_node = predict_next_node_with_bias(model, current_node, node_to_idx, idx_to_node, graph, visited, goal)

        # Handle dead ends or invalid predictions
        if next_node is None:
            if explore:
                neighbors = [n for n in graph.neighbors(current_node) if n not in visited]
                if neighbors:
                    weights = [1 / (manhattan_distance(n, goal) + 1e-6) for n in neighbors]
                    next_node = random.choices(neighbors, weights=weights, k=1)[0]  # Weighted random exploration
            if next_node is None:  # Backtrack with priority
                if not stack:
                    print("Dead end encountered. No more moves.")
                    break
                current_node = stack.pop(0)  # Backtrack to the highest-priority node
                continue

        # Add current node to stack, sorted by priority
        insort(stack, current_node, key=lambda n: manhattan_distance(n, goal))

        # Update path and move to the next node
        path.append(next_node)
        current_node = next_node

        if current_node == goal:
            print("Goal reached!")
            break

    # Refine the path and return it
    return refine_path(path, graph)


def refine_path(path, graph):
    """
    Refine the simulated path by removing unnecessary loops.
    """
    refined_path = []
    visited = set()

    for i, node in enumerate(path):
        if node not in visited:
            refined_path.append(node)
            visited.add(node)
        elif i > 0 and node in graph.neighbors(refined_path[-1]):
            # Preserve valid loops that connect to the last node
            refined_path.append(node)

    # Additional pruning: remove redundant connections
    pruned_path = []
    for i, node in enumerate(refined_path):
        if i == 0 or i == len(refined_path) - 1 or refined_path[i - 1] in graph.neighbors(node):
            pruned_path.append(node)

    return pruned_path


def compute_path_metrics(path, goal, optimal_path):
    """
    Compute metrics for the given path.
    """
    path_length = len(path)
    optimal_length = len(optimal_path)
    backtracking = len(path) - len(set(path))  # Nodes revisited

    return {
        "path_length": path_length,
        "optimal_length": optimal_length,
        "goal_reached": path[-1] == goal,
        "efficiency": optimal_length / path_length if path_length > 0 else 0,
        "backtracking": backtracking,
    }
