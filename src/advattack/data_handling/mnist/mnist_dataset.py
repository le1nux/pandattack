#!/usr/bin/env python3

import torch
import os
from typing import List
from advattack.error_handling.exception import DatasetNotFoundError
from advattack.data_handling.dataset import Dataset
import codecs
import numpy as np
import matplotlib.pyplot as plt
import glob
from advattack.util.logger import logger

class MNISTDataset(Dataset):
    """ MNIST dataset (http://yann.lecun.com/exdb/mnist/)
    """

    # TODO: Fix this ugly hard coding ...
    logger = logger.getChild("data_handling.dataset.mnist.mnist_dataset.MNISTDataset")

    def __init__(self, root_path, feature_transform_fun=None, target_transform_fun=None):
        super(MNISTDataset, self).__init__(root_path, feature_transform_fun, target_transform_fun)

    def __len__(self):
        return self.samples.shape[0]

    def visualize_samples(self, min_index, max_index):
        plots_per_column = 15 # we might want to make this configurable
        plot_count = max_index - min_index + 1
        cols = min(plot_count, plots_per_column)
        rows = int(plot_count/plots_per_column) + 1
        index = min_index
        plt.figure(figsize=(plots_per_column, rows))
        plt.subplots_adjust(top=0.8, bottom=0, hspace=1, wspace=0.5)
        for fig_index, index in enumerate(range(min_index, max_index+1)):
            ax = plt.subplot(rows, cols, fig_index+1)
            ax.set_axis_off()
            pixels, label = self[index]
            # pixels = self.samples[index]
            # label = self.labels[index]
            plt.title(f'idx:{index}\nlbl:{label}')
            plt.imshow(pixels, cmap='gray')
        #plt.tight_layout()
        plt.show()

    @classmethod
    def get_config(cls):
        # URL: (download_path, final_file_name, extraction_function)
        resource_extraction_mapping = {
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz': (
                "samples/", MNISTDataset.extract_samples),
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz': (
                "labels/", MNISTDataset.extract_labels),
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz': (
                "samples/", MNISTDataset.extract_samples),
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz': (
                "labels/", MNISTDataset.extract_labels)
        }
        return resource_extraction_mapping

    @classmethod
    def load(cls, path, feature_transform_fun=None, target_transform_fun=None):
        if cls.check_exists(path):
            return MNISTDataset(root_path=path)
        else:
            raise DatasetNotFoundError(f"Dataset not found in {path}.")

    @classmethod
    def check_exists(cls, root_path) -> bool:
        files = glob.glob(os.path.join(root_path, "**/*.pt"))
        return len(files) == 4

    @classmethod
    def create_dataset(cls, root_path) -> str:
        dataset_path = os.path.join(root_path, "mnist")
        resource_extraction_mapping = cls.get_config()
        cls.populate_data_repository(dataset_path, resource_extraction_mapping, force_download=True)
        return dataset_path

    @classmethod
    def extract_labels(cls, source_path):
        target_file_name = os.path.splitext(os.path.basename(source_path))[0] + ".pt"
        target_path = os.path.join(os.path.dirname(source_path), target_file_name)
        unzipped_path = Dataset.extract_gzip(source_path, remove_finished=True)
        data = cls.read_label_file(unzipped_path)
        with open(target_path, 'wb') as f:
            torch.save(data, f)
        os.remove(unzipped_path)

    @classmethod
    def extract_samples(cls, source_path):
        target_file_name = os.path.splitext(os.path.basename(source_path))[0] + ".pt"
        target_path = os.path.join(os.path.dirname(source_path), target_file_name)
        unzipped_path = Dataset.extract_gzip(source_path, remove_finished=True)
        data = MNISTDataset.read_image_file(unzipped_path)
        with open(target_path, 'wb') as f:
            torch.save(data, f)
        os.remove(unzipped_path)

    def save(self, path):
        pass

    def load_data_from_disc(self) -> (List, List):
        """Method that implements loading functionality of an on disk dataset.

        :param folder_path: Path to dataset folder
        :param data_files:
        :return: samples, targets
        """
        samples_paths = sorted(glob.glob(os.path.join(self.root_path, "samples/*.pt")))
        labels_paths = sorted(glob.glob(os.path.join(self.root_path, "labels/*.pt")))
        MNISTDataset.logger.debug(f"Loading samples from {samples_paths}")
        samples = [torch.load(path) for path in samples_paths]
        samples_tensor = torch.cat(samples, 0)

        labels = [torch.load(path) for path in labels_paths]
        labels_tensor = torch.cat(labels, 0)

        return samples_tensor, labels_tensor


    # Helper methods to extract the data from the provided raw dataset
    @classmethod
    def get_int(cls, b):
        return int(codecs.encode(b, 'hex'), 16)

    @classmethod
    def read_label_file(cls, path:str):
        with open(path, 'rb') as f:
            data = f.read()
            assert cls.get_int(data[:4]) == 2049
            length = cls.get_int(data[4:8])
            parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
            torch_tensor = torch.from_numpy(parsed).view(length).long()
            torch_tensor = torch_tensor.long()
            return torch_tensor


    @classmethod
    def read_image_file(cls, path:str):
        with open(path, 'rb') as f:
            data = f.read()
            print(cls.get_int(data[:4]))
            assert cls.get_int(data[:4]) == 2051
            length = cls.get_int(data[4:8])
            num_rows = cls.get_int(data[8:12])
            num_cols = cls.get_int(data[12:16])
            parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
            torch_tensor = torch.from_numpy(parsed).view(length, num_rows, num_cols)
            torch_tensor = torch_tensor.float()
            return torch_tensor

if __name__== "__main__":
    from advattack import datasets_path

    path = os.path.join(datasets_path, "mnist")
    if not MNISTDataset.check_exists(path) or True:
        path = MNISTDataset.create_dataset(root_path=datasets_path)

    dataset = MNISTDataset.load(path)
    dataset.visualize_samples(min_index=0, max_index=5)
    len(dataset)


