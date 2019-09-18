#!/usr/bin/env python3
import pytest
import tempfile
from advattack.data_handling.mnist.mnist_dataset import MNISTDataset
import shutil
from advattack.data_handling.dataset_loader import DatasetLoader
from torch.utils.data import BatchSampler, SubsetRandomSampler

class TestMNISTDataset:

    @pytest.fixture
    def tmp_dir(self):
        dir = tempfile.mkdtemp()
        yield dir
        shutil.rmtree(dir)

    @pytest.fixture
    def mnist_dataset_path(self, tmp_dir):
        dataset_path = MNISTDataset.create_dataset(root_path=tmp_dir)
        yield dataset_path
        shutil.rmtree(dataset_path)

    @pytest.fixture
    def mnist_dataset(self, mnist_dataset_path):
        dataset = MNISTDataset.load(mnist_dataset_path)
        yield dataset

    def test_dataset_loader_length(self, mnist_dataset):
        train_indices, valid_indices = mnist_dataset.get_train_and_validation_set_indices(train_valid_split_ratio=0.8, seed=2)
        train_loader = DatasetLoader(mnist_dataset, batch_sampler=BatchSampler(sampler=SubsetRandomSampler(train_indices),
                                                                         batch_size=50, drop_last=False))
        valid_loader = DatasetLoader(mnist_dataset, batch_sampler=BatchSampler(sampler=SubsetRandomSampler(valid_indices),
                                                                         batch_size=50, drop_last=False))
        assert (len(train_loader) == 56000) and (len(valid_loader) == 14000)
