import networkx as nx
import numpy as np

def maze_to_weighted_graph(maze, start, goal):
    """
    Convert a maze into a weighted graph where weights represent the shortest path to the goal.
    
    Parameters:
    - maze: 2D list representing the maze (e.g., '#' for walls, '.' for open paths)
    - start: Tuple (row, col) indicating the start position
    - goal: Tuple (row, col) indicating the goal position
    
    Returns:
    - G: A NetworkX graph with nodes and weighted edges
    """
    rows, cols = len(maze), len(maze[0])
    G = nx.Graph()
    
    # Add nodes for open cells
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] != '#':  # Open cell
                G.add_node((r, c), label=f"({r}, {c})")
    
    # Add edges between adjacent open cells
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] != '#':  # Open cell
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] != '#':
                        G.add_edge((r, c), (nr, nc), weight=1)  # Default weight 1
    
    # Compute shortest paths from all nodes to the goal
    shortest_paths = nx.single_source_dijkstra_path_length(G, goal)
    
    # Assign shortest path weights to the nodes
    for node in G.nodes:
        G.nodes[node]['shortest_path_to_goal'] = shortest_paths.get(node, np.inf)  # Inf if unreachable
    
    return G

# Example Maze
maze = [
    ["#", ".", "#", ".", ".", ".", "."],
    [".", ".", "#", "#", ".", "#", "#"],
    [".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", "#"],
    [".", ".", "S", "#", ".", ".", "."],
    [".", ".", ".", ".", "#", ".", "#"],
    ["#", "#", ".", ".", ".", ".", "."],
    [".", ".", ".", "#", ".", "#", "G"],
    ["#", ".", ".", "#", ".", ".", "#"],
    [".", "#", ".", "#", "#", ".", "."],
    [".", ".", ".", ".", ".", "#", "."]
]

start = (5, 2)  # 'S'
goal = (8, 6)   # 'G'

# Convert maze to a weighted graph
weighted_graph = maze_to_weighted_graph(maze, start, goal)

# Example: Print nodes with their shortest path to the goal
for node, data in weighted_graph.nodes(data=True):
    print(f"Node {node}, Shortest Path to Goal: {data['shortest_path_to_goal']}")

# Visualize the graph
import matplotlib.pyplot as plt

pos = nx.spring_layout(weighted_graph, seed=42)
node_labels = {node: f"{node}\n{data['shortest_path_to_goal']}" for node, data in weighted_graph.nodes(data=True)}

plt.figure(figsize=(10, 10))
nx.draw(weighted_graph, pos, with_labels=False, node_size=500, node_color='lightblue', edge_color='gray')
nx.draw_networkx_labels(weighted_graph, pos, labels=node_labels, font_size=8, font_color='black')
plt.title("Maze as Weighted Graph")
plt.show()
