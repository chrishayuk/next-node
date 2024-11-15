# train.py
import json
import os
import torch
from models import NodePredictor
from train_utils import train_model

os.makedirs("saved_models", exist_ok=True)

def load_dataset(file_path):
    """
    Load dataset from a JSONL file and extract 'current' and 'next' pairs.
    """
    with open(file_path, "r") as f:
        data = [json.loads(line.strip()) for line in f]

    training_data = []
    for item in data:
        if "training_data" in item:
            training_data.extend([(d["current"], d["next"]) for d in item["training_data"]])
    return training_data


def main():
    # Load dataset
    dataset_path = "datasets/maze/training/maze_training_data.jsonl"
    training_data = load_dataset(dataset_path)

    # Map nodes to indices
    node_to_idx = {}
    idx = 0
    for current, next_ in training_data:
        current_tuple = tuple(current)  # Convert to tuple
        next_tuple = tuple(next_)      # Convert to tuple

        if current_tuple not in node_to_idx:
            node_to_idx[current_tuple] = idx
            idx += 1
        if next_tuple not in node_to_idx:
            node_to_idx[next_tuple] = idx
            idx += 1

    training_indices = [
        (node_to_idx[tuple(curr)], node_to_idx[tuple(next_)]) for curr, next_ in training_data
    ]

    # Train model
    model = NodePredictor(len(node_to_idx), embedding_dim=16)
    train_model(model, training_indices)

    # Save model and node mappings
    torch.save(model.state_dict(), "saved_models/maze_model.pth")
    torch.save(node_to_idx, "saved_models/node_to_idx.pth")
    print("Model trained and saved.")


if __name__ == "__main__":
    main()
