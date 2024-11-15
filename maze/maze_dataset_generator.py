# maze/maze_dataset_generator.py
import os
import random
import argparse
import networkx as nx
import json
import logging
from tqdm import tqdm
from maze_generator import generate_random_maze
from maze_to_graph_converter import maze_to_graph
from graph_to_json import graph_to_json

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Ensure the datasets directory exists
os.makedirs("datasets/maze/training", exist_ok=True)

class MazeDatasetGenerator:
    def __init__(self, num_mazes, min_size, max_size, output_path):
        self.num_mazes = num_mazes
        self.min_size = min_size
        self.max_size = max_size
        self.output_path = output_path

    def validate_maze_connectivity(self, graph, start, goal):
        """
        Ensure there is a valid path between start and goal.
        """
        if not nx.has_path(graph, start, goal):
            raise ValueError(f"No path exists between start {start} and goal {goal}")

    def generate_context_embedding(self, maze):
        """
        Generate a dummy embedding for the maze context representation.
        Replace this with a model-based embedding if needed.
        """
        random.seed(hash(str(maze)))
        return [round(random.uniform(0.1, 2.0), 2) for _ in range(4)]

    def generate_single_maze(self, maze_id):
        """
        Generate a single maze dataset entry.
        """
        max_retries = 5
        for retry in range(max_retries):
            try:
                # Generate random maze
                rows = random.randint(self.min_size, self.max_size)
                cols = random.randint(self.min_size, self.max_size)
                maze, start, goal = generate_random_maze(rows, cols)

                # Convert maze to graph
                graph = maze_to_graph(maze)

                # Validate connectivity
                self.validate_maze_connectivity(graph, start, goal)

                # Compute shortest path
                path = nx.shortest_path(graph, source=start, target=goal)
                path_cost = nx.shortest_path_length(graph, source=start, target=goal, weight='weight')

                # Convert graph to JSON format with embeddings and weights
                graph_json = graph_to_json(graph, include_embeddings=True, compute_weights=True)

                # Prepare dataset entry
                dataset_entry = {
                    "context_representation": maze,
                    "graph": graph_json,
                    "context_embedding": self.generate_context_embedding(maze),
                    "path": [str(node) for node in path],
                    "path_cost": round(path_cost, 2),
                }

                logger.info(f"Generated maze {maze_id}: Size ({rows}x{cols}), Start: {start}, Goal: {goal}")
                return dataset_entry
            except (ValueError, nx.NetworkXNoPath, nx.NodeNotFound) as e:
                logger.warning(f"Retrying maze {maze_id} ({retry + 1}/{max_retries}) due to error: {e}")
        raise RuntimeError(f"Failed to generate maze {maze_id} after {max_retries} retries")

    def generate_maze_dataset(self):
        """
        Generate a dataset of mazes with randomized start and goal positions.
        """
        with open(self.output_path, "w") as f:
            for i in tqdm(range(1, self.num_mazes + 1), desc="Generating mazes"):
                try:
                    dataset_entry = self.generate_single_maze(i)
                    # Write each JSON object followed by a newline
                    f.write(json.dumps(dataset_entry) + "\n")
                except RuntimeError as e:
                    logger.error(f"Skipping maze {i} due to repeated errors: {e}")



def main(num_mazes, min_size, max_size):
    """
    Main function to generate maze datasets.
    """
    # Set the output path
    output_path = "datasets/maze/training/maze_training_data.jsonl"

    # Setup the dataset generator
    generator = MazeDatasetGenerator(num_mazes, min_size, max_size, output_path)

    # Generate data
    generator.generate_maze_dataset()


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Generate random maze datasets for training.")

    # Setup arguments
    parser.add_argument("--num-mazes", type=int, default=10, help="Number of mazes to generate.")
    parser.add_argument("--min-size", type=int, default=5, help="Minimum size of the maze (rows/cols).")
    parser.add_argument("--max-size", type=int, default=10, help="Maximum size of the maze (rows/cols).")

    # Parse
    args = parser.parse_args()

    # Generate
    main(args.num_mazes, args.min_size, args.max_size)
