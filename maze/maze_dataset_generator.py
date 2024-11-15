# maze/maze_dataset_generator.py
import os
import random
import argparse
import networkx as nx
import json
from maze_generator import generate_random_maze
from maze_to_graph_converter import maze_to_graph

# Ensure the datasets directory exists
os.makedirs("datasets", exist_ok=True)

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

    def generate_training_data(self, graph, start, goal):
        """
        Generate shortest-path training data for the maze.
        """
        path = nx.shortest_path(graph, source=start, target=goal)
        training_data = [{"current": path[i], "next": path[i + 1]} for i in range(len(path) - 1)]
        return training_data

    def generate_maze_dataset(self):
        """
        Generate a dataset of mazes with randomized start and goal positions.
        """
        with open(self.output_path, "w") as f:
            for i in range(self.num_mazes):
                try:
                    # Generate random maze
                    rows = random.randint(self.min_size, self.max_size)
                    cols = random.randint(self.min_size, self.max_size)
                    maze, start, goal = generate_random_maze(rows, cols)

                    # Convert maze to graph
                    graph = maze_to_graph(maze)

                    # Validate connectivity
                    self.validate_maze_connectivity(graph, start, goal)

                    # Generate training data
                    training_data = self.generate_training_data(graph, start, goal)

                    # Save dataset entry
                    dataset_entry = {
                        "maze": maze,
                        "start": start,
                        "goal": goal,
                        "graph": nx.node_link_data(graph),
                        "training_data": training_data,
                    }
                    f.write(json.dumps(dataset_entry) + "\n")
                    print(f"Generated maze {i + 1}/{self.num_mazes}: Size ({rows}x{cols}), Start: {start}, Goal: {goal}")
                except (ValueError, nx.NetworkXNoPath, nx.NodeNotFound) as e:
                    print(f"Skipping maze {i + 1} due to error: {e}")

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
