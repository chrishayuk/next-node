# Maze

## Generating a Maze

```bash
python maze/maze_dataset_generator.py --num-mazes 1000 --min-size 6 --max-size 12
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
