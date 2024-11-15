# dataset_generation/base_dataset_generator.py
class BaseDatasetGenerator:
    def generate_data(self, graph, start, goal):
        """
        Generate dataset for training.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")
