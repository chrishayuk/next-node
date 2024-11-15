import json
import random
import networkx as nx
import matplotlib.pyplot as plt

from maze.maze_generator import generate_random_maze

# # Load the maze from the JSONL file
# file_path = 'datasets/maze/training/maze_training_data.jsonl'

# # Parse the JSON object from the file
# with open(file_path, 'r') as f:
#     maze_data = json.load(f)

# Generate random maze
rows = 6
cols = 6
maze_data, start, goal = generate_random_maze(rows, cols)

# Create a graph from the maze data
maze_graph = nx.Graph()

# Add nodes and edges to the graph
for node in maze_data['graph']['nodes']:
    maze_graph.add_node(tuple(node['id']))

for link in maze_data['graph']['links']:
    source = tuple(link['source'])
    target = tuple(link['target'])
    maze_graph.add_edge(source, target)

# Visualize the graph
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(maze_graph, seed=42)  # Use spring layout for visualization
nx.draw(
    maze_graph,
    pos,
    with_labels=True,
    node_size=500,
    node_color='lightblue',
    edge_color='gray',
    font_size=10,
    font_color='black'
)
plt.title("Maze Graph Visualization")
plt.show()

