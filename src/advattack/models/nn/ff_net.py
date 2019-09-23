import torch
import torch.nn as nn
import torch.nn.functional as F
from advattack.models.nn.net import NNModel
from typing import List

class FFModel(NNModel):
    def __init__(self, layers:List[int]):
        super(FFModel, self).__init__()
        self.layers = layers
        # create fully connected layers
        self.fc_layers = nn.ModuleList([nn.Linear(input_size, layers[i+1]) for i, input_size in enumerate(layers[:-1])])
        self.relu = torch.nn.ReLU()

    def step(self, samples):
        output = self.fc_layers[0](samples)
        for layer in self.fc_layers[1:-1]:
            output = layer(output)
            output = self.relu(output)
        output = self.fc_layers[-1](output)
        output = self.relu(output)
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
        config["layers"] = self.layers
        return config



