# maze/dataset_validator.py
import json
import argparse

def validate_json_format(entry):
    """
    Validate the format of a single JSON entry.
    """
    try:
        # Validate context_representation
        if not isinstance(entry.get("context_representation"), list):
            return "Invalid or missing 'context_representation'."
        for row in entry["context_representation"]:
            if not isinstance(row, list) or not all(isinstance(cell, str) for cell in row):
                return "Invalid 'context_representation' structure."

        # Validate graph
        graph = entry.get("graph")
        if not isinstance(graph, dict):
            return "Invalid or missing 'graph'."
        
        # Validate nodes
        nodes = graph.get("nodes")
        if not isinstance(nodes, list):
            return "Invalid or missing 'nodes' in 'graph'."
        for node in nodes:
            if not all(key in node for key in ["id", "label", "embedding"]):
                return "Missing keys in a node."
            if not isinstance(node["id"], str) or not isinstance(node["label"], str):
                return "Invalid 'id' or 'label' in a node."
            if not (isinstance(node["embedding"], list) and len(node["embedding"]) == 3 and
                    all(isinstance(x, (float, int)) for x in node["embedding"])):
                return "Invalid 'embedding' in a node."

        # Validate edges
        edges = graph.get("edges")
        if not isinstance(edges, list):
            return "Invalid or missing 'edges' in 'graph'."
        for edge in edges:
            if not all(key in edge for key in ["source", "target", "weight"]):
                return "Missing keys in an edge."
            if not isinstance(edge["source"], str) or not isinstance(edge["target"], str):
                return "Invalid 'source' or 'target' in an edge."
            if not isinstance(edge["weight"], (float, int)):
                return "Invalid 'weight' in an edge."

        # Validate context_embedding
        context_embedding = entry.get("context_embedding")
        if not (isinstance(context_embedding, list) and len(context_embedding) == 4 and
                all(isinstance(x, (float, int)) for x in context_embedding)):
            return "Invalid or missing 'context_embedding'."

        # Validate path
        path = entry.get("path")
        if not (isinstance(path, list) and all(isinstance(node_id, str) for node_id in path)):
            return "Invalid or missing 'path'."

        # Validate path_cost
        path_cost = entry.get("path_cost")
        if not isinstance(path_cost, (float, int)):
            return "Invalid or missing 'path_cost'."

        return "Valid JSON format."
    except Exception as e:
        return f"Error during validation: {e}"


def validate_jsonl_file(jsonl_file):
    """
    Validate all entries in a JSONL file.
    """
    with open(jsonl_file, "r") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                entry = json.loads(line)
                validation_result = validate_json_format(entry)
                if validation_result != "Valid JSON format.":
                    print(f"Line {line_number}: {validation_result}")
            except json.JSONDecodeError as e:
                print(f"Line {line_number}: Invalid JSON - {e}")

    # complete
    print("Validation Completed")

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Validate the format of a JSONL file.")
    parser.add_argument(
        "--jsonl-file",
        type=str,
        required=True,
        help="Path to the JSONL file to validate."
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate JSONL file
    validate_jsonl_file(args.jsonl_file)
