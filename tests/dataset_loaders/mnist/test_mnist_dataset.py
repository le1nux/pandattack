#!/usr/bin/env python3
import pytest
import tempfile
import os
from advattack.data_handling.mnist.mnist_dataset import MNISTDataset
import shutil
from advattack.error_handling.exception import DatasetNotFoundError
import glob
import torch

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

    @pytest.fixture
    def mnist_dataset(self, mnist_dataset_path):
        dataset = MNISTDataset.load(mnist_dataset_path)
        yield dataset

    def test_create_dataset(self, tmp_dir):
        dataset_path = MNISTDataset.create_dataset(root_path=tmp_dir)
        files = glob.glob(os.path.join(dataset_path, "**/*.pt"))
        assert len(files) == 4

    def test_check_exists_when_not_exist(self, tmp_dir):
        assert not MNISTDataset.check_exists(tmp_dir)

    def test_check_exists_when_exists(self, mnist_dataset_path):
        assert MNISTDataset.check_exists(mnist_dataset_path)

    def test_load_not_existing_dataset(self, tmp_dir):
        threw_error = False
        try:
            MNISTDataset.load(tmp_dir)
        except DatasetNotFoundError:
            threw_error = True
        assert threw_error

    def test_load_existing_dataset(self, mnist_dataset_path):
        dataset = MNISTDataset.load(mnist_dataset_path)
        assert len(dataset) == 70000

    def test___len__(self, mnist_dataset):
        assert len(mnist_dataset) == 70000

    def test_ieteration_via___getitem__(self, mnist_dataset):
        def check_format(pixels, label):
            assert pixels.shape == torch.Size([28, 28])

        for pixels, label in mnist_dataset:
            check_format(pixels, label)

    def test_out_of_bounds_via___getitem__(self, mnist_dataset):
        length = len(mnist_dataset)
        threw_error = False
        try:
            mnist_dataset[length]
        except Exception as e:
            threw_error = True
        assert threw_error






