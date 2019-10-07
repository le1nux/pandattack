import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, SubsetRandomSampler
from advattack.data_handling.dataset_repository import DatasetRepository
from advattack.data_handling.cifar.cifar_dataset import Cifar10Dataset
from advattack.data_handling.dataset_loader import DatasetLoader
from advattack.models.nn.conv_net import ConvNet
from advattack.models.model_repository import ModelRepository
from torchvision import transforms
from advattack.util.tensorboard import TensorboardWrapper
from advattack.util.tensorboard import TensorboardMode


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))

batch_size = 100
learning_rate = 0.001
epochs = 150

dataset_class = Cifar10Dataset
model_class = ConvNet

# instantiate model
tensorboard_writer = TensorboardWrapper.get_summary_writer(dataset_identifier=dataset_class.get_dataset_identifier(), model_identifier=model_class.get_model_identifier(), mode=TensorboardMode.TRAIN)

model_config = \
    {
        "layer_config":
            [
                {
                    "type": "conv",
                    "params": {"in_channels": 3, "out_channels": 32, "kernel_size": 3, "stride": 1}
                }, {
                    "type": "conv",
                    "params": {"in_channels": 32, "out_channels": 64, "kernel_size": 3, "stride": 1}
                },{
                    "type": "fc",
                    "params": {"in_channels": 12544, "out_channels": 10}
                }
            ],
        "tensorboard_writer": tensorboard_writer
    }

loss_function = nn.NLLLoss()
model = model_class(**model_config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# generate training set
feature_transform_fun = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# load dataset
dataset = DatasetRepository.get_dataset(dataset_class, feature_transform_fun=feature_transform_fun)
train_indices, valid_indices = dataset.get_train_and_validation_set_indices(train_valid_split_ratio=0.8, seed=2)
train_loader = DatasetLoader(dataset,
                             batch_sampler=BatchSampler(sampler=SubsetRandomSampler(train_indices),
                                                        batch_size=batch_size,
                                                        drop_last=False),
                             collate_fn=lambda batch: DatasetLoader.square_matrix_collate_fn(batch, channels=3))
valid_loader = DatasetLoader(dataset, batch_sampler=BatchSampler(sampler=SubsetRandomSampler(valid_indices),
                                                                 batch_size=batch_size,
                                                                 drop_last=False),
                             collate_fn=lambda batch: DatasetLoader.square_matrix_collate_fn(batch, channels=3))
# train model
model.train_model(train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer, loss_function=loss_function, epochs=epochs)

# save model to disk
ModelRepository.store_model(model=model, dataset_class=dataset_class)

# load model from disk
model = ModelRepository.get_model(model_class, dataset_class=dataset_class)

