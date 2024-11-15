from datasets import load_dataset, Dataset, DatasetDict
from typing import Union
import os


class DatasetLoader:
    """
    A class for loading datasets from Hugging Face Hub or locally.
    Supports datasets like CNN/DailyMail, FEVER, SQuAD, and XSUM.
    """

    def __init__(self, local_cache_dir: str = r"C:\Users\batti\llm-evaluation\data"):
        """
        Initialize the DatasetLoader with a local cache directory.
        
        Args:
            local_cache_dir (str): Directory to store/load datasets locally.
        """
        self.local_cache_dir = os.path.abspath(local_cache_dir)
        os.makedirs(self.local_cache_dir, exist_ok=True)

    def load_dataset(self, dataset_name: str, split: str = "test") -> Union[Dataset, DatasetDict]:
        """
        Load a dataset from Hugging Face Hub or from the local cache.

        Args:
            dataset_name (str): Name of the dataset (e.g., "cnn_dailymail", "squad", "xsum", "fever").
            split (str): Dataset split to load (e.g., "train", "validation", "test"). Defaults to "test".

        Returns:
            Union[Dataset, DatasetDict]: The loaded dataset.

        Raises:
            ValueError: If the dataset name or split is invalid.
            FileNotFoundError: If the dataset is not found locally and cannot be downloaded.
            ConnectionError: If there is a problem connecting to Hugging Face Hub.
        """
        try:
            print(f"Attempting to load dataset: {dataset_name}, split: {split}")
            dataset_path = os.path.join(self.local_cache_dir, dataset_name)

            # Check if dataset exists locally
            if os.path.exists(dataset_path):
                print(f"Loading dataset locally from: {dataset_path}")
                return load_dataset(path=dataset_path, split=split)

            # Otherwise, try loading from Hugging Face Hub
            print(f"Dataset not found locally. Attempting to download {dataset_name} from Hugging Face Hub...")
            return load_dataset(dataset_name, split=split, cache_dir=self.local_cache_dir)

        except FileNotFoundError as fnf_error:
            print(f"Error: Dataset '{dataset_name}' not found locally or remotely.")
            raise FileNotFoundError(
                f"Dataset '{dataset_name}' not available locally or on Hugging Face Hub."
            ) from fnf_error

        except ConnectionError as conn_error:
            print(f"Error: Unable to connect to Hugging Face Hub to download '{dataset_name}'.")
            raise ConnectionError(
                f"Failed to download dataset '{dataset_name}' due to connectivity issues."
            ) from conn_error

        except ValueError as val_error:
            print(f"Error: Invalid dataset name or split for '{dataset_name}'.")
            raise ValueError(f"Dataset '{dataset_name}' or split '{split}' is invalid.") from val_error

        except Exception as e:
            print(f"Unexpected error while loading dataset '{dataset_name}': {e}")
            raise RuntimeError(
                f"An unexpected error occurred while loading dataset '{dataset_name}'."
            ) from e
        

if __name__ == "__main__":
    loader = DatasetLoader()
    dataset = loader.load_dataset("EdinburghNLP/xsum", split="test")
    print(dataset)
