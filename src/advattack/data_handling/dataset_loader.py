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
        if collate_fn is None:
            collate_fn = self.standard_collate_fn
        super(DatasetLoader, self).__init__(dataset=dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

    def __len__(self):
        return len(self.batch_sampler) * self.batch_sampler.batch_size

    def standard_collate_fn(self, batch):
        # batch contains a list of tuples of structure (sequence, target)
        inputs = [item[0] for item in batch]
        inputs = torch.stack(inputs)
        inputs = inputs.view(inputs.shape[0], -1)
        inputs_len = inputs.shape[0]
        targets_tensor = torch.tensor([item[1] for item in batch]).to(inputs[0].device)
        # mapping = [item[2] for item in batch]
        return [inputs, inputs_len, targets_tensor]


if __name__ == "__main__":
    mnist_path = os.path.join(datasets_path, "mnist")
    dataset = MNISTDataset.load(mnist_path)
    train_indices, valid_indices = dataset.get_train_and_validation_set_indices(train_valid_split_ratio=0.8, seed=2)
    train_loader = DatasetLoader(dataset, batch_sampler=BatchSampler(sampler=SubsetRandomSampler(train_indices),
                                                                  batch_size=50, drop_last=False))
    valid_loader = DatasetLoader(dataset, batch_sampler=BatchSampler(sampler=SubsetRandomSampler(valid_indices),
                                                                  batch_size=50, drop_last=False))

    print(f"length of training set: {len(train_loader)}, length of validation set: {len(valid_loader)}")