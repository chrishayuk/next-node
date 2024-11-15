# maze/maze_to_graph_converter.py
import networkx as nx
import argparse
import json
from maze_generator import generate_random_maze, print_maze

def calculate_weight(node1, node2, label1, label2):
    """
    Calculate the weight of an edge based on node labels.
    - Regular walkable nodes have a weight of 1.0.
    - Start and goal nodes are treated with normal weights.
    - Walls have an infinite weight.
    """
    if label1 == 'wall' or label2 == 'wall':
        return float('inf')  # Walls are impassable
    return 1.0  # Default weight for walkable paths


def maze_to_graph(maze, label_map=None):
    """
    Convert a maze represented as a grid into a graph representation with labeled nodes and weighted edges.
    Args:
        maze (list of list of str): 2D maze grid with characters representing cells.
        label_map (dict, optional): Mapping of maze symbols to labels (e.g., {'S': 'start', '.': 'walkable', 'G': 'goal'}).
                                    Defaults to None, in which case symbols are used as labels directly.
    Returns:
        networkx.Graph: A graph representing the maze.
    """
    if label_map is None:
        label_map = {
            'S': 'start',
            '.': 'walkable',
            'G': 'goal',
            '#': 'wall',
        }

    G = nx.Graph()
    rows, cols = len(maze), len(maze[0])

    # Add nodes and edges
    for r in range(rows):
        for c in range(cols):
            cell = maze[r][c]
            node_label = label_map.get(cell, cell)
            G.add_node((r, c), label=node_label)

            # Skip walls for edge connections
            if cell == '#':
                continue

            # Connect to top neighbor
            if r > 0:
                neighbor = maze[r - 1][c]
                neighbor_label = label_map.get(neighbor, neighbor)
                weight = calculate_weight((r, c), (r - 1, c), node_label, neighbor_label)
                G.add_edge((r, c), (r - 1, c), weight=weight)

            # Connect to left neighbor
            if c > 0:
                neighbor = maze[r][c - 1]
                neighbor_label = label_map.get(neighbor, neighbor)
                weight = calculate_weight((r, c), (r, c - 1), node_label, neighbor_label)
                G.add_edge((r, c), (r, c - 1), weight=weight)

    return G


def print_graph(graph):
    """
    Print the graph for debugging purposes, including edge weights.
    """
    print("\nGraph Nodes:")
    for node, data in graph.nodes(data=True):
        print(f"Node: {node}, Label: {data['label']}")

    print("\nGraph Edges:")
    for source, target, data in graph.edges(data=True):
        weight = "∞" if data["weight"] == float('inf') else data["weight"]
        print(f"Edge: {source} -> {target}, Weight: {weight}")


def load_maze_from_json(file_path):
    """
    Load a maze from a JSON file.
    Args:
        file_path (str): Path to the JSON file.
    Returns:
        list: Maze represented as a 2D list of strings.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['maze']


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Convert a maze to a graph.")
    parser.add_argument(
        "--json-file", type=str, help="Path to a JSON file containing the maze."
    )
    args = parser.parse_args()

    # Load maze from JSON or generate a random maze
    if args.json_file:
        maze = load_maze_from_json(args.json_file)
        print("Loaded Maze from JSON:")
    else:
        maze, start, goal = generate_random_maze(5, 5, wall_prob=0.3)
        print("Generated Maze:")

    # Print the maze
    print_maze(maze)

    # Convert the maze to a graph
    graph = maze_to_graph(maze)

    # Print the graph nodes and edges
    print_graph(graph)

