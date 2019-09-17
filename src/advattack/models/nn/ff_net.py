import torch
import torch.nn as nn
import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.data import DataLoader, BatchSampler, SubsetRandomSampler
import torch.nn.functional as F
from advattack.dataset_loaders.mnist.mnist_dataset import MNISTDataset
from advattack import datasets_path, tensorboard_path, models_path
from torchvision import transforms


class NNModel(nn.Module):
    def __init__(self, loss_function, use_tensorboard=False):
        super(NNModel, self).__init__()
        self.loss_function = loss_function
        if use_tensorboard:
            self.tb_writer_train, self.tb_writer_valid = self.set_up_tensorboard()

    def set_up_tensorboard(self):
        # set up tensorboard
        tb_dir_train = os.path.join(tensorboard_path, "/ff/train/")
        tb_dir_valid = os.path.join(tensorboard_path, "/ff/valid/")
        os.makedirs(os.path.dirname(tb_dir_train), exist_ok=True)
        os.makedirs(os.path.dirname(tb_dir_valid), exist_ok=True)
        tb_writer_train = SummaryWriter(log_dir=tb_dir_train, flush_secs=10)
        tb_writer_valid = SummaryWriter(log_dir=tb_dir_valid, flush_secs=10)
        return tb_writer_train, tb_writer_valid

    def train_model(self, train_loader, valid_loader, optimizer, epochs=1):
        print("Starting Training loss:")
        self.evaluate_model(train_loader, 0)
        print("Starting Validation loss:")
        self.evaluate_model(valid_loader, 0)
        print("\n=================================================================================================\n")

        for epoch in tqdm.tqdm(range(epochs)):  # again, normally you would NOT do 300 epochs, it is toy data
            self.train_epoch(train_loader, loss_function, optimizer)
            print("Training loss:")
            self.evaluate_model(data_loader=train_loader, epoch=epoch)
            print("Validation loss:")
            self.evaluate_model(data_loader=valid_loader, epoch=epoch)
            print("\n============================================================================================\n")
            # model_save_path = os.path.join(f'../../models/lstm_model/epoch_{epoch}.pt')
            # model.save(model_save_path)

    def evaluate_model(self, data_loader, epoch):
        test_loss = 0
        correct = 0
        for samples, batch_size, targets in data_loader:
            with torch.no_grad():
                outputs = self(samples).squeeze(1)
                test_loss += F.nll_loss(outputs, targets, reduction='sum').item()  # sum up batch loss
                pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(targets.view_as(pred)).sum().item()
        accuracy = correct / len(data_loader.dataset)
        print(f'Average loss: {test_loss}, Accuracy: {accuracy}')
        return test_loss

        #self.tb_writer.add_scalar('Loss', total_loss, epoch)


class FFModel(NNModel):
    def __init__(self, input_size, num_layers, hiden_size_fc, output_size, loss_function):
        super(FFModel, self).__init__(loss_function)
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size_fc = hiden_size_fc
        self.output_size = output_size
        # input layer
        self.fc_layers = [nn.Linear(self.input_size, self.hidden_size_fc)]
        # add hidden layers
        self.fc_layers = self.fc_layers + [nn.Linear(self.hidden_size_fc, self.hidden_size_fc) for _ in range(num_layers)]
        # output layer
        self.out = nn.Linear(self.hidden_size_fc, output_size)
        self.relu = torch.nn.ReLU()

    def step(self, samples):
        output = self.fc_layers[0](samples)
        for layer in self.fc_layers[1:]:
            output = layer(output)
            output = self.relu(output)
        output = self.out(output)
        output = self.relu(output)
        output = F.log_softmax(output, dim=1)
        return output

    def forward(self, samples):
        output = self.step(samples)
        return output

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print("Model saved.")

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


def my_collate_fn(batch):
    # batch contains a list of tuples of structure (sequence, target)
    inputs = [item[0] for item in batch]
    inputs = torch.stack(inputs)
    inputs = inputs.view(inputs.shape[0], -1)
    inputs_len = inputs.shape[0]
    targets_tensor = torch.tensor([item[1] for item in batch]).to(inputs[0].device)
    # mapping = [item[2] for item in batch]
    return [inputs, inputs_len, targets_tensor]

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: " + str(device))

    num_features = 28*28
    num_layers = 2
    hidden_size_fc = 120
    output_size = 10

    batch_size = 30
    learning_rate = 0.003

    # instantiate model
    loss_function = nn.NLLLoss()
    model = FFModel(input_size=num_features, num_layers=num_layers, hiden_size_fc=hidden_size_fc, output_size=output_size, loss_function=loss_function).to(device)
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
    train_loader = DataLoader(dataset, batch_sampler=BatchSampler(sampler=SubsetRandomSampler(train_indices), batch_size=batch_size, drop_last=False), collate_fn=my_collate_fn)
    valid_loader = DataLoader(dataset, batch_sampler=BatchSampler(sampler=SubsetRandomSampler(valid_indices), batch_size=batch_size, drop_last=False), collate_fn=my_collate_fn)

    model.train_model(train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer, epochs=5)

