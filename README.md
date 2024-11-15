# Maze

## Generating a Maze
the following generates a random maze, and prints out the result

```bash
python maze/maze_generator.py
```

if you want to save as json for testing later

```bash
python maze/maze_generator.py --save-path "datasets/maze/testing/generated_maze.json"
```

## Visualizing a Maze
I've include some simple tools to be able to visualize maze inference test files

if you want to visualize a random maze:

```bash
python maze/maze_visualizer.py --json-file "datasets/maze/testing/generated_maze.json"
```

if you want to visualize an existing test file maze

```bash
python maze/maze_dataset_visualizer.py
```

## Generating a Graph of a Maze
the following generates a random maze, and converts to a graph

```bash
python maze/maze_to_graph_converter.py
```

or

```bash
python maze/maze_to_graph_converter.py --json-file "datasets/maze/testing/generated_maze.json"
```

## Generating a Maze Dataset
The following will generate a maze dataset for training.

```bash
python maze/maze_dataset_generator.py --num-mazes 1 --min-size 6 --max-size 12
```

### Validating a Generated Maze Dataset
The following will validate a maze dataset for training.

```bash
python maze/dataset_validator.py --jsonl-file datasets/maze/training/maze_training_data.jsonl
```

### Visualizing Maze Datasets
I've include some simple tools to be able to visualize maze training files

```bash
python maze/maze_dataset_visualizer.py --jsonl-file datasets/maze/training/maze_training_data.jsonl
```
