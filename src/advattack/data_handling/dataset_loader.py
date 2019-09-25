import torch
from torch.utils.data import DataLoader
from advattack.data_handling.dataset import Dataset


class DatasetLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_sampler=None, collate_fn=None):
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

    def get_dataset_identifier(self) -> str:
        return self.dataset.get_dataset_identifier()
