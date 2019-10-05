import torch
from advattack.error_handling.exception import AttackError, AdversarialNotFoundError
from advattack.models.nn.ff_net import FFNet
import torch.nn as nn
from advattack.data_handling.mnist.mnist_dataset import MNISTDataset
from advattack.data_handling.dataset_repository import DatasetRepository
from torchvision import transforms
from advattack.data_handling.dataset_loader import DatasetLoader
from advattack.models.model_repository import ModelRepository
from advattack.attacks.white_box.fgsm import FGSM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))

# instantiate model
loss_function = nn.NLLLoss()
model = ModelRepository.get_model(FFNet, MNISTDataset)

# load data set
feature_transform_fun = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset_class = MNISTDataset
dataset = DatasetRepository.get_dataset(dataset_class, feature_transform_fun=feature_transform_fun)
dataset_loader = DatasetLoader(dataset)

fgsm = FGSM(model=model, epsilon=1, device=device)
iterator = iter(dataset_loader)
for idx, (original_image, _, original_target) in enumerate(iterator):
    print(idx)
    try:
        attack_result = fgsm.attack(original_image, original_target)
        attack_result.visualize()
    except AdversarialNotFoundError:
        pass
    except AttackError:
        pass