import torch
import torch.nn as nn
import tqdm
import os
import torch.nn.functional as F
import shutil
from advattack.data_handling.dataset_loader import DatasetLoader
from torch.utils.tensorboard import SummaryWriter


class NNModel(nn.Module):
    def __init__(self, tensorboard_writer: SummaryWriter, seed: int = None):
        super(NNModel, self).__init__()
        self.tensorboard_writer = tensorboard_writer
        if seed is not None:
            torch.manual_seed(seed)

    def train_model(self, train_loader: DatasetLoader, valid_loader: DatasetLoader, optimizer, loss_function, epochs=1):
        print("Starting Training loss:")
        self.evaluate_model(train_loader, 0, dataset_split_tag="train")
        print("Starting Validation loss:")
        self.evaluate_model(valid_loader, 0, dataset_split_tag="validation")
        print("\n===============================================================================================\n")

        for epoch in tqdm.tqdm(range(epochs)):  # again, normally you would NOT do 300 epochs, it is toy data
            self.train_epoch(train_loader, loss_function, optimizer)
            print("Training loss:")
            self.evaluate_model(data_loader=train_loader, epoch=epoch, dataset_split_tag="train")
            print("Validation loss:")
            self.evaluate_model(data_loader=valid_loader, epoch=epoch, dataset_split_tag="validation")
            print("\n============================================================================================\n")
            # model_save_path = os.path.join(f'../../models/lstm_model/epoch_{epoch}.pt')
            # model.save(model_save_path)

    def evaluate_model(self, data_loader: DatasetLoader, epoch, dataset_split_tag):
        test_loss = 0
        correct = 0
        for samples, batch_size, targets in data_loader:
            with torch.no_grad():
                outputs = self(samples).squeeze(1)
                test_loss += F.nll_loss(outputs, targets, reduction='sum').item()  # sum up batch loss
                pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(targets.view_as(pred)).sum().item()

        accuracy = correct / len(data_loader)
        test_loss = test_loss / len(data_loader)
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar('Loss/' + dataset_split_tag, test_loss, epoch)
            self.tensorboard_writer.add_scalar('Accuracy/' + dataset_split_tag, accuracy, epoch)
        print(f'Average loss: {test_loss}, Accuracy: {accuracy}')
        return test_loss

    def save_model(self, folder_path) -> str:
        shutil.rmtree(folder_path, ignore_errors=True)
        os.makedirs(folder_path)
        full_path = os.path.join(folder_path, "model.pt")
        torch.save(self.state_dict(), full_path)
        return full_path

    def load_model(self, folder_path):
        full_path = os.path.join(folder_path, "model.pt")
        self.load_state_dict(torch.load(full_path))
        self.eval()

    @classmethod
    def get_model_identifier(cls):
        return cls.__mro__[0].__name__

    def get_config(self):
        return dict()
