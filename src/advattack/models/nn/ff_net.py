import torch
import torch.nn as nn
import os
from torch.utils.data import BatchSampler, SubsetRandomSampler
import torch.nn.functional as F
from advattack.data_handling.mnist.mnist_dataset import MNISTDataset
from advattack import datasets_path
from advattack.data_handling.dataset_loader import DatasetLoader
from advattack.models.nn.net import NNModel
from torchvision import transforms
import numpy as np


class FFModel(NNModel):
    def __init__(self, layers, loss_function):
        super(FFModel, self).__init__(loss_function)
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

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: " + str(device))

    batch_size = 100
    learning_rate = 0.0008
    epochs = 30

    # instantiate model
    layers = np.array([28*28, 250, 250, 250, 10]).flatten()
    loss_function = nn.NLLLoss()
    model = FFModel(layers, loss_function=loss_function).to(device)
    print(model._type())
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # generate training set
    feature_transform_fun = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist_path = os.path.join(datasets_path, "mnist")
    dataset = MNISTDataset.load(mnist_path, feature_transform_fun=feature_transform_fun)
    train_indices, valid_indices = dataset.get_train_and_validation_set_indices(train_valid_split_ratio=0.8, seed=2)
    train_loader = DatasetLoader(dataset, batch_sampler=BatchSampler(sampler=SubsetRandomSampler(train_indices), batch_size=batch_size, drop_last=False))
    valid_loader = DatasetLoader(dataset, batch_sampler=BatchSampler(sampler=SubsetRandomSampler(valid_indices), batch_size=batch_size, drop_last=False))

    model.train_model(train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer, epochs=epochs)

    model.save_model()

