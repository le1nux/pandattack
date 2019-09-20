#!/usr/bin/env python3
import pytest
import tempfile
import shutil
from advattack.models.nn.ff_net import  FFModel
import torch
import torch.nn as nn
import os
from torch.utils.data import BatchSampler, SubsetRandomSampler
from advattack.data_handling.mnist.mnist_dataset import MNISTDataset
from advattack import datasets_path
from advattack.data_handling.dataset_loader import DatasetLoader
from torchvision import transforms
import numpy as np

class TestFFModel:

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

    @pytest.fixture
    def train_valid_loader(self, mnist_dataset):
        batch_size = 30

        # generate training set
        feature_transform_fun = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_indices, valid_indices = mnist_dataset.get_train_and_validation_set_indices(train_valid_split_ratio=0.8, seed=2)
        train_loader = DatasetLoader(mnist_dataset, batch_sampler=BatchSampler(sampler=SubsetRandomSampler(train_indices),
                                                                         batch_size=batch_size, drop_last=False))
        valid_loader = DatasetLoader(mnist_dataset, batch_sampler=BatchSampler(sampler=SubsetRandomSampler(valid_indices),
                                                                         batch_size=batch_size, drop_last=False))
        yield train_loader, valid_loader

    @pytest.fixture
    def ff_net_model(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: " + str(device))

        layers = np.array([28 * 28, 250, 250, 250, 10]).flatten()

        learning_rate = 0.2
        epochs = 1

        # instantiate model
        loss_function = nn.NLLLoss()
        model = FFModel(layers, loss_function=loss_function).to(device)
        yield model

    def test_train_model(self, train_valid_loader, ff_net_model):
        # To test if training is working we store the initial model weights and
        # compare them to the ones calculated after one epoch. If they are different
        # we assume model is learning something. Fully training a model is not feasible
        # for testing in a CI environment.

        train_loader, valid_loader = train_valid_loader

        initial_module_dict = ff_net_model.fc_layers._modules
        init_weight_dict = {}
        for layer_key, module in initial_module_dict.items():
            init_weight_dict[layer_key] = initial_module_dict["0"].weight.data.clone()

        optimizer = torch.optim.SGD(ff_net_model.parameters(), lr=0.1)
        ff_net_model.train_model(train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer, epochs=1)

        trained_module_dict = ff_net_model.fc_layers._modules
        weights_different = True
        for layer_key, module in ff_net_model.fc_layers._modules.items():
            initial_weights = init_weight_dict[layer_key]
            trained_weights = trained_module_dict[layer_key].weight.data
            if np.array_equal(initial_weights, trained_weights):
                weights_different = False
                break
        assert weights_different









