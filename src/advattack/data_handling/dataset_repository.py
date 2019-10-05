from advattack.data_handling.dataset import Dataset
import os
from advattack import datasets_path


class DatasetRepository:

    @staticmethod
    def get_dataset(dataset_class, feature_transform_fun=None, target_transform_fun=None) -> Dataset:
        root_path = os.path.join(datasets_path, dataset_class.get_dataset_identifier())
        if not dataset_class.check_exists(root_path):
            dataset_class.create_dataset(root_path)
        dataset = dataset_class.load(root_path, feature_transform_fun, target_transform_fun)
        return dataset