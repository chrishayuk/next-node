# maze_utils.py
import random
import networkx as nx

def maze_to_graph(maze):
    G = nx.Graph()
    rows, cols = len(maze), len(maze[0])
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] != '#':  # Open cell
                G.add_node((r, c))
                if r > 0 and maze[r-1][c] != '#':  # Connect to top neighbor
                    G.add_edge((r, c), (r-1, c))
                if c > 0 and maze[r][c-1] != '#':  # Connect to left neighbor
                    G.add_edge((r, c), (r, c-1))
    return G

def visualize_solution(maze, solution_path):
    maze_copy = [row[:] for row in maze]
    for r, c in solution_path:
        if maze_copy[r][c] not in ['S', 'G']:
            maze_copy[r][c] = '*'
    for row in maze_copy:
        print(' '.join(row))

def generate_random_maze(rows, cols, start='S', goal='G', wall_prob=0.3):
    """ Generates a random maze """
    # creates am empty grid, . indicates empty
    maze = [['.' for _ in range(cols)] for _ in range(rows)]

    # specifies start position and goal position
    maze[0][0] = start
    maze[-1][-1] = goal

    # loop though each row and column
    for r in range(rows):
        for c in range(cols):
            # if not the start or the goal, randomly generate a wall based on probability
            if maze[r][c] not in [start, goal] and random.random() < wall_prob:
                # congrats square, you're now a wall
                maze[r][c] = '#'

    # return the maze
    return maze
