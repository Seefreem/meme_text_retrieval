from datasets import Dataset, load_dataset
from src.datasets.memecap_dataset import memecap_dataset

def get_meme_text_dataloader(dataset_name: str, meme_shape: tuple):
    """
    Load a dataset of the given name.

    Args:
    dataset_name (str): Support datasets: memecap.
    meme_shape (tuple): The shape of the meme, if the width or height is negative, then the memes will be kept as unchanged.
    """
    if dataset_name == 'memecap':
        return memecap_dataset(meme_shape[0], meme_shape[1])
    else:
        print(f"Error: no supporting for dataset: {dataset_name} ")
        return None

