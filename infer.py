import argparse
from maze_utils import maze_to_graph, visualize_solution
from train_utils import simulate_path_with_pruning_and_exploration
from models import NodePredictor
import torch
import json


def load_maze(file_path):
    """
    Load a maze from a JSON file.
    """
    with open(file_path, 'r') as f:
        maze = json.load(f)
    return maze


def main():
    parser = argparse.ArgumentParser(description="Infer paths in mazes using a trained model.")
    parser.add_argument('--maze', type=str, required=True, help="Path to the maze JSON file.")
    args = parser.parse_args()

    # Load maze
    maze = load_maze(args.maze)
    graph = maze_to_graph(maze)
    start = next((r, c) for r, row in enumerate(maze) for c, cell in enumerate(row) if cell == 'S')
    goal = next((r, c) for r, row in enumerate(maze) for c, cell in enumerate(row) if cell == 'G')

    # Load saved mappings and model state_dict
    saved_node_to_idx = torch.load("saved_models/node_to_idx.pth", weights_only=True)
    saved_state_dict = torch.load("saved_models/maze_model.pth", weights_only=True)

    # Create new mappings for the unseen maze
    new_node_to_idx = {node: i for i, node in enumerate(graph.nodes)}
    new_idx_to_node = {i: node for node, i in new_node_to_idx.items()}

    # Initialize model for unseen maze
    new_num_nodes = len(graph.nodes)
    embedding_dim = 16  # Matches training setup
    model = NodePredictor(new_num_nodes, embedding_dim)

    # Adjust the state_dict to load compatible weights
    updated_state_dict = model.state_dict()
    for name, param in saved_state_dict.items():
        if name in updated_state_dict and param.size() == updated_state_dict[name].size():
            updated_state_dict[name] = param
    model.load_state_dict(updated_state_dict)
    model.eval()

    # Simulate path with pruning and goal bias
    path = simulate_path_with_pruning_and_exploration(model, start, goal, new_node_to_idx, new_idx_to_node, graph)
    print("Simulated Path:", path)

    # Visualize the solution
    visualize_solution(maze, path)


if __name__ == "__main__":
    main()
