# maze/maze_generator.py
import random

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


def generate_random_maze(rows, cols, wall_prob=0.3):
    """
    Generates a random maze with walls and walkable paths.
    """
    maze = [['#' if random.random() < wall_prob else '.' for _ in range(cols)] for _ in range(rows)]
    maze, start, goal = randomize_start_and_goal(maze)

    return maze, start, goal
