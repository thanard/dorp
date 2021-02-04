from .numpy_datasets.npy_dataset import NPYDataset
from .numpy_datasets.npy_dataset_load import NPYDatasetLoad

def get_dataset_class(dataset):
    dataset_mappings = {
        'NPYDataset': 'NPYDataset',
        'NPYDatasetLoad': 'NPYDatasetLoad'
    }
    dataset_class = dataset_mappings.get(dataset, dataset)
    dataset_class = globals().get(dataset_class)
    return dataset_class
