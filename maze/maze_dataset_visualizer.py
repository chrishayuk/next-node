
# maze/maze_dataset_visualizer.py
import json
import matplotlib.pyplot as plt
import argparse
from colorama import Fore, Style, Back


def visualize_maze_plot(maze):
    """
    Visualize a maze as a binary grid using matplotlib.
    """
    grid = [[0 if cell == '.' else 1 for cell in row] for row in maze]
    plt.imshow(grid, cmap='binary', interpolation='nearest')
    plt.axis('off')
    plt.show()


def visualize_maze_ascii(maze):
    """
    Visualize a maze as ASCII art with colored start and goal.
    """
    ascii_maze = "\n".join(
        "".join(
            Fore.GREEN + cell + Style.RESET_ALL if cell == "S" else
            Fore.RED + cell + Style.RESET_ALL if cell == "G" else cell
            for cell in row
        )
        for row in maze
    )
    print(ascii_maze)


def visualize_maze_ascii_color(maze):
    """
    Visualize a maze as ASCII art with colored walls, start, goal, and dots for paths.
    """
    color_mapping = {
        "#": Back.RED + Fore.WHITE + "#" + Style.RESET_ALL,   # Walls (White on Red background)
        ".": Fore.WHITE + "." + Style.RESET_ALL,             # Open path (White foreground)
        "S": Back.BLUE + Fore.WHITE + "S" + Style.RESET_ALL, # Start (White on Blue background)
        "G": Back.GREEN + Fore.WHITE + "G" + Style.RESET_ALL # Goal (White on Green background)
    }
    ascii_maze = "\n".join(
        "".join(color_mapping.get(cell, cell) for cell in row)
        for row in maze
    )
    print(ascii_maze)


def process_jsonl_file(dataset_path, num_mazes, mode):
    """
    Process a JSONL file and visualize mazes.
    """
    with open(dataset_path, "r") as f:
        for i, line in enumerate(f):
            if i >= num_mazes:
                break
            data = json.loads(line)
            print(f"Maze {i+1}: Start: {data['start']}, Goal: {data['goal']}")
            visualize_maze(data["maze"], mode)


def process_json_file(dataset_path, mode):
    """
    Process a JSON file and visualize the single maze.
    """
    with open(dataset_path, "r") as f:
        maze = json.load(f)
        visualize_maze(maze, mode)


def visualize_maze(maze, mode):
    """
    Visualize a single maze in the specified mode.
    """
    if mode == "ascii":
        visualize_maze_ascii(maze)
    elif mode == "ascii-color":
        visualize_maze_ascii_color(maze)
    elif mode == "plot":
        visualize_maze_plot(maze)
    else:
        print(f"Invalid mode: {mode}. Please choose 'ascii', 'ascii-color', or 'plot'.")


def main(dataset_path, num_mazes, mode):
    """
    Determine the dataset type and visualize mazes.
    """
    try:
        if dataset_path.endswith(".jsonl"):
            process_jsonl_file(dataset_path, num_mazes, mode)
        elif dataset_path.endswith(".json"):
            process_json_file(dataset_path, mode)
        else:
            print("Unsupported file format. Please provide a .jsonl or .json file.")
    except FileNotFoundError:
        print(f"Error: File '{dataset_path}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file '{dataset_path}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Visualize mazes from JSON or JSONL datasets.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset file (.jsonl or .json) containing mazes."
    )
    parser.add_argument(
        "--num-mazes",
        type=int,
        default=5,
        help="Number of mazes to visualize from a JSONL file. Ignored for JSON files."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ascii", "ascii-color", "plot"],
        default="ascii-color",
        help="Visualization mode: 'ascii', 'ascii-color', or 'plot'. Default is 'ascii-color'."
    )
    args = parser.parse_args()

    # Run main function
    main(args.dataset_path, args.num_mazes, args.mode)
