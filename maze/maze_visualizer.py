# maze/maze_visualizer.py
import json
import matplotlib.pyplot as plt
import argparse
from colorama import Fore, Style, Back
from maze.maze_generator import generate_random_maze, print_maze

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
    Visualize a maze as ASCII art with colored walls, start, goal, and paths.
    """
    color_mapping = {
        "#": Back.RED + Fore.WHITE + "#" + Style.RESET_ALL,   # Walls
        ".": Fore.WHITE + "." + Style.RESET_ALL,             # Open path
        "S": Back.BLUE + Fore.WHITE + "S" + Style.RESET_ALL, # Start
        "G": Back.GREEN + Fore.WHITE + "G" + Style.RESET_ALL # Goal
    }
    ascii_maze = "\n".join(
        "".join(color_mapping.get(cell, cell) for cell in row)
        for row in maze
    )
    print(ascii_maze)


def visualize_maze_plot(maze):
    """
    Visualize a maze as a binary grid using matplotlib.
    """
    grid = [[0 if cell == '.' else 1 for cell in row] for row in maze]
    plt.imshow(grid, cmap='binary', interpolation='nearest')
    plt.axis('off')
    plt.show()


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


def load_maze_from_json(file_path):
    """
    Load a maze from a JSON file.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["maze"]


def main(json_file=None, mode="ascii-color"):
    """
    Main function to visualize a maze from a file or randomly generate one.
    """

    # check if a random maze
       
    if json_file:
        print(f"Loading maze from {json_file}...")
        maze = load_maze_from_json(json_file)
    else:
        print("Generating a random maze...")
        maze, _, _ = generate_random_maze(5, 5, wall_prob=0.3)
        print("Generated Maze:")
        print_maze(maze)

    print("\nVisualizing Maze:")
    visualize_maze(maze, mode)


if __name__ == "__main__":
    # setup the parser
    parser = argparse.ArgumentParser(description="Visualize mazes from JSON files or generate random mazes.")

    # setup the arguments
    parser.add_argument(
        "--json-file",
        type=str,
        help="Path to the JSON file containing the maze."
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

    # visualize
    main(json_file=args.json_file, mode=args.mode)
