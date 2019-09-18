from torch.utils.data import DataLoader
import torch
import os
from torch.utils.data import DataLoader, BatchSampler, SubsetRandomSampler
from advattack.data_handling.mnist.mnist_dataset import MNISTDataset
from advattack import datasets_path

class DatasetLoader(DataLoader):

    def __init__(self, dataset, batch_sampler=None, collate_fn=None):
        """

        :param dataset:
        :param batch_size:
        :param shuffle:
        :param batch_sampler:
        :param collate_fn: if non it will use the default collate function
        """
        super(DatasetLoader, self).__init__(dataset=dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

    def __len__(self):
        return len(self.batch_sampler) * self.batch_sampler.batch_size



if __name__ == "__main__":
    mnist_path = os.path.join(datasets_path, "mnist")

    dataset = MNISTDataset.load(mnist_path)
    train_indices, valid_indices = dataset.get_train_and_validation_set_indices(train_valid_split_ratio=0.8, seed=2)
    train_loader = DatasetLoader(dataset, batch_sampler=BatchSampler(sampler=SubsetRandomSampler(train_indices),
                                                                  batch_size=50, drop_last=False))
    valid_loader = DatasetLoader(dataset, batch_sampler=BatchSampler(sampler=SubsetRandomSampler(valid_indices),
                                                                  batch_size=50, drop_last=False))

    print(f"length of training set: {len(train_loader)}, length of validation set: {len(valid_loader)}")