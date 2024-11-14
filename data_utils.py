import networkx as nx

from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra_path_length

def generate_paths(G, start, goal):
    return nx.shortest_path(G, source=start, target=goal)

def generate_goal_biased_training_data(G, start, goal):
    distances = single_source_dijkstra_path_length(G, goal)
    path = generate_paths(G, start, goal)
    training_data = []

    for i in range(len(path) - 1):
        current_node = path[i]
        next_node = path[i + 1]
        score = distances.get(next_node, float('inf'))  # Distance to goal
        training_data.append((current_node, next_node, score))

    training_data.sort(key=lambda x: x[2])  # Sort by proximity to goal
    return [(curr, next_) for curr, next_, _ in training_data]
