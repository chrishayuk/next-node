import json
import argparse
from random import sample
from maze_visualizer import visualize_maze, visualize_maze_ascii

def visualize_solution_ascii(maze, solution_path):
    """
    Visualize a maze as ASCII art with the solution path.
    """
    # copy the maze
    maze_copy = [row[:] for row in maze]

    # sow the solution
    for r, c in solution_path:
        if maze_copy[r][c] not in ['S', 'G']:
            maze_copy[r][c] = '*'

    # visualize
    visualize_maze_ascii(maze_copy)
    
def process_jsonl_file(dataset_path, num_mazes, mode):
    """
    Process a JSONL file and visualize mazes.
    """
    with open(dataset_path, "r") as f:
        data_lines = f.readlines()

    # Randomly select mazes if requested
    selected_lines = sample(data_lines, min(num_mazes, len(data_lines)))

    for i, line in enumerate(selected_lines, start=1):
        try:
            data = json.loads(line)
            maze = data["context_representation"]
            solution_path = [eval(node) for node in data.get("path", [])]
            print(f"\nMaze {i}: Path Cost: {data.get('path_cost', 'N/A')}")
            visualize_maze(maze, mode)
            if solution_path:
                print("Visualizing solution:")
                visualize_solution_ascii(maze, solution_path)
        except KeyError as e:
            print(f"Skipping maze {i} due to missing key: {e}")


def main(jsonl_file, num_mazes, mode):
    """
    Determine the dataset type and visualize mazes.
    """
    try:
        if jsonl_file.endswith(".jsonl"):
            process_jsonl_file(jsonl_file, num_mazes, mode)
        else:
            print("Unsupported file format. Please provide a .jsonl file.")
    except FileNotFoundError:
        print(f"Error: File '{jsonl_file}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file '{jsonl_file}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # setup parser
    parser = argparse.ArgumentParser(description="Visualize mazes from JSONL datasets.")

    # parse arguments
    parser.add_argument(
        "--jsonl-file",
        type=str,
        required=True,
        help="Path to the dataset file (.jsonl) containing mazes."
    )
    parser.add_argument(
        "--num-mazes",
        type=int,
        default=5,
        help="Number of mazes to visualize."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ascii", "ascii-color", "plot"],
        default="ascii-color",
        help="Visualization mode: 'ascii', 'ascii-color', or 'plot'. Default is 'ascii-color'."
    )

    # parse
    args = parser.parse_args()

    # Run main function
    main(args.jsonl_file, args.num_mazes, args.mode)
