import torch.nn as nn
import torch.nn.functional as F
from advattack.models.nn.net import NNModel
from typing import List
from torch.utils.tensorboard import SummaryWriter


class FFModel(NNModel):
    def __init__(self, layer_config: List[int], tensorboard_writer: SummaryWriter = None, seed: int = None):
        super(FFModel, self).__init__(tensorboard_writer=tensorboard_writer, seed=seed)
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

    def train_epoch(self, train_loader, loss_function, optimizer):
        for samples, batch_size, targets in train_loader:
            # sequence = sequence.view(1, -1, num_features)
            # targets = targets.view(-1, 1)
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            self.zero_grad()
            # Step 3. Run our forward pass.
            predictions = self(samples).squeeze(1)
           # prediction = prediction.view(-1, 1)
            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(predictions, targets)
            loss.backward()
            optimizer.step()

    def get_config(self):
        config = super().get_config()
        config["layers"] = self.layer_config
        return config



