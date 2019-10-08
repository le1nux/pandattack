#!/usr/bin/env python3
import pytest
import tempfile
import shutil
from advattack.models.nn.conv_net import ConvNet
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, SubsetRandomSampler
from advattack.data_handling.mnist.mnist_dataset import MNISTDataset
from advattack.data_handling.dataset_loader import DatasetLoader
from torchvision import transforms
import numpy as np


class TestConvNet:

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

        # generate training set
        feature_transform_fun = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset = MNISTDataset.load(mnist_dataset_path, feature_transform_fun=feature_transform_fun)
        yield dataset

    @pytest.fixture
    def train_valid_loader(self, mnist_dataset):
        batch_size = 100

        train_indices, valid_indices = mnist_dataset.get_train_and_validation_set_indices(train_valid_split_ratio=0.8, seed=2)
        train_loader = DatasetLoader(mnist_dataset,
                                     batch_sampler=BatchSampler(sampler=SubsetRandomSampler(train_indices),
                                                                batch_size=batch_size,
                                                                drop_last=False),
                                     collate_fn=DatasetLoader.square_matrix_collate_fn)
        valid_loader = DatasetLoader(mnist_dataset,
                                     batch_sampler=BatchSampler(sampler=SubsetRandomSampler(valid_indices),
                                                                batch_size=batch_size,
                                                                drop_last=False),
                                     collate_fn=DatasetLoader.square_matrix_collate_fn)
        yield train_loader, valid_loader

    @pytest.fixture()
    def device(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        yield device

    @pytest.fixture
    def conv_net_model(self, device):
        print("Device: " + str(device))

        model_config = \
            {
                "layer_config":
                    [
                        {
                            "type": "conv",
                            "params": {"in_channels": 1, "out_channels": 32, "kernel_size": 3, "stride": 1}
                        }, {
                        "type": "conv",
                        "params": {"in_channels": 32, "out_channels": 64, "kernel_size": 3, "stride": 1}
                    }, {
                        "type": "fc",
                        "params": {"in_channels": 9216, "out_channels": 10}
                    }
                    ]
            }
        # instantiate model
        model = ConvNet(**model_config).to(device)
        yield model

    def get_weight_dict(self, layer_module_dict):
        weight_dict = {}
        for layer_key, module in layer_module_dict.items():
            weight_dict[layer_key] = layer_module_dict["0"].weight.data.clone()
        return weight_dict

    def check_weight_inequality(self, init_weights, trained_weights):
        weights_different = True
        for layer_key, module in init_weights.items():
            if np.array_equal(init_weights[layer_key], trained_weights[layer_key]):
                weights_different = False
                break
        return weights_different

    def test_train_model(self, train_valid_loader, conv_net_model, device):
        # To test if training is working we store the initial model weights and
        # compare them to the ones calculated after one epoch. If they are different
        # we assume model is learning something. Fully training a model is not feasible
        # for testing in a CI environment.
        loss_function = nn.NLLLoss()

        train_loader, valid_loader = train_valid_loader

        init_fc_weight_dict = self.get_weight_dict(conv_net_model.fc_layers._modules)
        init_conv_weight_dict = self.get_weight_dict(conv_net_model.conv_layers._modules)

        optimizer = torch.optim.Adam(conv_net_model.parameters(), lr=0.01)
        conv_net_model.train_model(train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer, loss_function=loss_function, epochs=1, device=device)

        trained_fc_weight_dict = self.get_weight_dict(conv_net_model.fc_layers._modules)
        trained_conv_weight_dict = self.get_weight_dict(conv_net_model.conv_layers._modules)

        assert self.check_weight_inequality(init_fc_weight_dict, trained_fc_weight_dict)
        assert self.check_weight_inequality(init_conv_weight_dict, trained_conv_weight_dict)

