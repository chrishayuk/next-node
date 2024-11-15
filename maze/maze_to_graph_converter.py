# maze/maze_to_graph_converter.py
import networkx as nx

def maze_to_graph(maze):
    """
    Convert a maze represented as a grid into a graph representation.
    """
    G = nx.Graph()
    rows, cols = len(maze), len(maze[0])
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] != '#':  # Open cell
                G.add_node((r, c))
                if r > 0 and maze[r - 1][c] != '#':  # Connect to top neighbor
                    G.add_edge((r, c), (r - 1, c))
                if c > 0 and maze[r][c - 1] != '#':  # Connect to left neighbor
                    G.add_edge((r, c), (r, c - 1))
    return G