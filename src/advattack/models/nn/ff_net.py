import torch.nn as nn
import torch.nn.functional as F
from advattack.models.nn.net import NNModel
from typing import List
from torch.utils.tensorboard import SummaryWriter


class FFNet(NNModel):
    def __init__(self, layer_config: List[int], tensorboard_writer: SummaryWriter = None, seed: int = None):
        super(FFNet, self).__init__(tensorboard_writer=tensorboard_writer, seed=seed)
        self.layer_config = layer_config
        # create fully connected layers
        self.fc_layers = nn.ModuleList([nn.Linear(input_size, layer_config[i + 1]) for i, input_size in enumerate(layer_config[:-1])])

    def step(self, samples):
        output = samples
        for layer in self.fc_layers:
            output = layer(output)
            output = F.relu(output)
        output = F.log_softmax(output, dim=1)
        return output

    def forward(self, samples):
        output = self.step(samples)
        return output

    def get_config(self):
        config = super().get_config()
        config["layer_config"] = self.layer_config
        return config



