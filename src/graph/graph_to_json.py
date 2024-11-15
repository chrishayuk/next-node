import json
import networkx as nx
import random


def generate_embedding(label):
    """
    Generate a deterministic embedding based on the label.
    Replace this with an actual embedding generator for meaningful embeddings.
    """
    random.seed(hash(label))
    return [round(random.uniform(0.1, 1.0), 3) for _ in range(3)]


def compute_edge_weight(source, target):
    """
    Compute edge weight based on Manhattan distance between nodes.
    """
    x1, y1 = eval(source)
    x2, y2 = eval(target)
    return round(abs(x1 - x2) + abs(y1 - y2), 1)


def graph_to_json(G, include_embeddings=True, compute_weights=False, add_edge_metadata=False):
    """
    Convert a NetworkX graph to a JSON-compatible structure.
    Args:
        G (networkx.Graph): The graph to convert.
        include_embeddings (bool): Whether to include embeddings for each node.
        compute_weights (bool): Whether to compute weights dynamically for edges.
        add_edge_metadata (bool): Whether to add additional metadata to edges.
    Returns:
        dict: JSON-compatible representation of the graph.
    """
    # Serialize nodes
    nodes = []
    for node, data in G.nodes(data=True):
        node_entry = {"id": str(node), "label": data.get("label", "")}
        if include_embeddings:
            node_entry["embedding"] = data.get("embedding", generate_embedding(node_entry["label"]))
        nodes.append(node_entry)

    # Serialize edges
    edges = []
    for u, v, data in G.edges(data=True):
        edge_entry = {
            "source": str(u),
            "target": str(v),
            "weight": compute_edge_weight(str(u), str(v)) if compute_weights else data.get("weight", 1.0),
        }
        if add_edge_metadata:
            edge_entry.update({
                "type": "shortest_path" if compute_weights else "default",
                "direction": "undirected"  # Adjust for directed graphs
            })
        edges.append(edge_entry)

    return {"nodes": nodes, "edges": edges}


# Example Usage
if __name__ == "__main__":
    # Create a simple graph
    G = nx.Graph()
    G.add_node((0, 0), label="S")
    G.add_node((0, 1), label=".")
    G.add_node((1, 1), label=".")
    G.add_node((2, 4), label="G")
    G.add_edge((0, 0), (0, 1))  # Default weight
    G.add_edge((0, 1), (1, 1))  # Default weight
    G.add_edge((1, 1), (2, 4))  # Default weight

    # Convert graph to JSON with additional metadata
    graph_json = graph_to_json(G, include_embeddings=True, compute_weights=True, add_edge_metadata=True)

    # Print the graph in JSON format
    print(json.dumps(graph_json, indent=4))
