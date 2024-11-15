# maze/maze_generator.py
import argparse
import random
import json
from collections import deque

def randomize_start_and_goal(maze):
    """
    Randomly place a start ('S') and goal ('G') in the maze.
    Ensure sufficient distance and walkable path between them.
    """
    rows, cols = len(maze), len(maze[0])
    walkable_positions = [(r, c) for r in range(rows) for c in range(cols) if maze[r][c] == '.']

    if len(walkable_positions) < 2:
        raise ValueError("Not enough walkable positions for both start and goal!")

    start = random.choice(walkable_positions)
    walkable_positions.remove(start)
    goal = random.choice(walkable_positions)

    maze[start[0]][start[1]] = 'S'
    maze[goal[0]][goal[1]] = 'G'

    return maze, start, goal


def is_path_exists(maze, start, goal):
    """
    Use BFS to check if there's a valid path between start and goal.
    """
    rows, cols = len(maze), len(maze[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    queue = deque([start])
    visited = set([start])

    while queue:
        r, c = queue.popleft()

        if (r, c) == goal:
            return True

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and maze[nr][nc] in ['.', 'G']:
                queue.append((nr, nc))
                visited.add((nr, nc))

    return False


def generate_random_maze(rows, cols, wall_prob=0.3, max_retries=10, save_path=None):
    """
    Generates a random maze with walls and walkable paths.
    Ensures a valid path exists between start and goal.
    If save_path is provided, saves the maze to a JSON file.
    """
    if rows < 3 or cols < 3:
        raise ValueError("Maze dimensions must be at least 3x3!")

    for _ in range(max_retries):
        maze = [['#' if random.random() < wall_prob else '.' for _ in range(cols)] for _ in range(rows)]
        maze, start, goal = randomize_start_and_goal(maze)

        if is_path_exists(maze, start, goal):
            if save_path:
                save_maze_to_json(maze, save_path)
            return maze, start, goal

    raise RuntimeError("Failed to generate a valid maze after multiple retries.")


def save_maze_to_json(maze, save_path):
    """
    Save the maze to a JSON file.
    """
    maze_data = {"maze": maze}
    with open(save_path, "w") as f:
        json.dump(maze_data, f, indent=4)
    print(f"Maze saved to {save_path}")


def print_maze(maze):
    """
    Print the maze for debugging purposes.
    """
    for row in maze:
        print(' '.join(row))
    print()


# Example usage
if __name__ == "__main__":
    # setup the parser
    parser = argparse.ArgumentParser(description="Generate a random maze.")

    # setup arguments
    parser.add_argument("--rows", type=int, default=5, help="Number of rows in the maze.")
    parser.add_argument("--cols", type=int, default=5, help="Number of columns in the maze.")
    parser.add_argument("--wall-prob", type=float, default=0.3, help="Probability of a wall in each cell.")
    parser.add_argument("--save-path", type=str, help="Path to save the generated maze as JSON.")

    # parse
    args = parser.parse_args()

    # generate the maze
    maze, start, goal = generate_random_maze(args.rows, args.cols, args.wall_prob, save_path=args.save_path)

    # print the maze
    print("Generated Maze:")
    print_maze(maze)
    print(f"Start: {start}, Goal: {goal}")
