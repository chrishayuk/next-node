# train.py
from maze_utils import maze_to_graph, generate_random_maze
from data_utils import generate_goal_biased_training_data
from models import NodePredictor
from train_utils import train_model
import torch
import os

os.makedirs("saved_models", exist_ok=True)

def main():
    maze = generate_random_maze(6, 6)
    graph = maze_to_graph(maze)
    start, goal = (0, 0), (5, 5)

    training_data = generate_goal_biased_training_data(graph, start, goal)
    node_to_idx = {node: i for i, node in enumerate(graph.nodes)}
    training_indices = [(node_to_idx[curr], node_to_idx[next_]) for curr, next_ in training_data]

    model = NodePredictor(len(graph.nodes), embedding_dim=16)
    train_model(model, training_indices)

    torch.save(model.state_dict(), "saved_models/maze_model.pth")
    torch.save(node_to_idx, "saved_models/node_to_idx.pth")
    print("Model trained and saved.")

if __name__ == "__main__":
    main()
