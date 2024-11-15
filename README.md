# Maze

## Generating a Maze
the following generates a random maze, and prints out the result

```bash
python src/maze/maze_generator.py
```

if you want to save as json for testing later

```bash
python src/maze/maze_generator.py --save-path "datasets/maze/testing/generated_maze.json"
```

## Generating a Graph of a Maze
the following generates a random maze, and converts to a graph

```bash
python src/maze/maze_to_graph_converter.py
```

or

```bash
python src/maze/maze_to_graph_converter.py --json-file "datasets/maze/testing/generated_maze.json"
```


## Generating a Maze Dataset

```bash
python src/maze/maze_dataset_generator.py --num-mazes 1000 --min-size 6 --max-size 12
```

## Visualizing Mazes
I've include some simple tools to be able to visualize maze inference test files and training files

### test files
run the following

```bash
python maze/maze_dataset_visualizer.py --dataset-path datasets/maze/testing/simple_maze.json
```

### training files
run the following

```bash
python maze/maze_dataset_visualizer.py --dataset-path datasets/maze/training/maze_training_data.jsonl
```
