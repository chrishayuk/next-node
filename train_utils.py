import torch


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
            # Manhattan distance to goal as a heuristic
            distance_to_goal = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
            neighbor_scores[neighbor] = next_probs[node_to_idx[neighbor]] / (distance_to_goal + 1e-6)  # Avoid division by zero

    if not neighbor_scores:
        return None  # Dead end
    return max(neighbor_scores, key=neighbor_scores.get)  # Neighbor with the highest adjusted score


def simulate_path_with_pruning_and_bias(model, start, goal, node_to_idx, idx_to_node, graph, max_steps=100):
    """
    Simulate a path from start to goal with pruning and goal bias.
    """
    current_node = start
    path = [current_node]
    visited = set()
    stack = []  # For backtracking

    for _ in range(max_steps):
        visited.add(current_node)
        next_node = predict_next_node_with_bias(model, current_node, node_to_idx, idx_to_node, graph, visited, goal)

        if next_node is None:  # Dead end, backtrack
            if not stack:
                print("Dead end encountered. No more moves.")
                break
            current_node = stack.pop()  # Backtrack to the previous node
        else:
            stack.append(current_node)  # Push the current node to the stack
            path.append(next_node)
            current_node = next_node

        if current_node == goal:
            print("Goal reached!")
            break

    # Prune the path
    pruned_path = []
    for node in path:
        if node not in pruned_path:
            pruned_path.append(node)

    return pruned_path


def simulate_path(model, start, goal, node_to_idx, idx_to_node, graph, max_steps=100):
    """
    Simulate a path from start to goal without pruning or goal bias (basic version).
    """
    current_node = start
    path = [current_node]
    visited = set()

    for _ in range(max_steps):
        visited.add(current_node)
        next_node = predict_next_node_with_bias(model, current_node, node_to_idx, idx_to_node, graph, visited, goal)

        if next_node is None:
            print("Dead end encountered. No more moves.")
            break

        path.append(next_node)
        current_node = next_node

        if current_node == goal:
            print("Goal reached!")
            break

    return path
