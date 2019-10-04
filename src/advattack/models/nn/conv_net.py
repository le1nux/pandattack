from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
from advattack.models.nn.net import NNModel
from torch.utils.tensorboard import SummaryWriter
from typing import List


class ConvNet(NNModel):
    def __init__(self, layer_config: List, tensorboard_writer: SummaryWriter = None, seed: int = None):
        super(ConvNet, self).__init__(tensorboard_writer=tensorboard_writer, seed=seed)
        self.layer_config = layer_config
        self.conv_layers = nn.ModuleList([])
        self.fc_layers = nn.ModuleList([])
        for layer in layer_config:
            if layer["type"] == "conv":
                self.conv_layers.append(ConvNet.create_conv_layer_from_config(layer["params"]))
            elif layer["type"] == "fc":
                self.fc_layers.append(ConvNet.create_fc_layer_from_config(layer["params"]))

    @staticmethod
    def create_conv_layer_from_config(layer_dict) -> nn.Module:
            return nn.Conv2d(in_channels=layer_dict["in_channels"],
                             out_channels=layer_dict["out_channels"],
                             kernel_size=layer_dict["kernel_size"],
                             stride=layer_dict["stride"])

    @staticmethod
    def create_fc_layer_from_config(layer_dict) -> nn.Module:
            return nn.Linear(in_features=layer_dict["in_channels"],
                             out_features=layer_dict["out_channels"])

    def forward(self, x):
        output = x
        # run through convolution layers
        for layer in self.conv_layers:
            output = layer(output)
            output = F.relu(output)
            # output = F.max_pool2d(output, 2, 2)
        # run through fully connected layers
        output = F.max_pool2d(output, 2, 2)
        # flatten output
        output = output.view(x.shape[0], -1)
        for layer in self.fc_layers:
            output = layer(output)
            output = F.relu(output)
        return F.log_softmax(output, dim=1)

    def train_epoch(self, train_loader, loss_function, optimizer):
        for samples, batch_size, targets in train_loader:
            self.zero_grad()
            predictions = self(samples).squeeze(1)
            loss = loss_function(predictions, targets)
            loss.backward()
            optimizer.step()

    def get_config(self):
        config = super().get_config()
        config["layer_config"] = self.layer_config
        return config
