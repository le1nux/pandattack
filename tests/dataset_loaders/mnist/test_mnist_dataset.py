#!/usr/bin/env python3
import pytest
import tempfile
import os
from advattack.dataset_loaders.mnist.mnist_dataset import MNISTDataset
import shutil
from advattack.error_handling.exception import DatasetError
import glob

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
        except DatasetError:
            threw_error = True
        assert threw_error

    def test_load_existing_dataset(self, mnist_dataset_path):
        path = MNISTDataset.create_dataset(root_path=mnist_dataset_path)
        dataset = MNISTDataset.load(path)
        dataset = MNISTDataset.load(mnist_dataset_path)




