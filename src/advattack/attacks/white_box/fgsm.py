from advattack.attacks.attack import  Attack, AttackResult
from advattack.models.nn.net import NNModel
import torch
import torch.nn.functional as F
from advattack.error_handling.exception import AttackError, AdversarialNotFoundError
from advattack.models.nn.ff_net import FFModel
import torch.nn as nn
from advattack.data_handling.mnist.mnist_dataset import MNISTDataset
import os
from advattack import datasets_path
from torchvision import transforms
from advattack.data_handling.dataset_loader import DatasetLoader
from advattack.models.model_repository import ModelRepository


class FGSM(Attack):
    def __init__(self, model: NNModel):
        super(FGSM, self).__init__(model)

    def calc_adversarial(self, original_image, epsilon, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        gradient_image = epsilon*sign_data_grad
        perturbed_image = original_image + gradient_image
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 255)
        # Return the perturbed image
        return perturbed_image, gradient_image

    def search_for_adversarial_example(self, original_image, original_target, epsilon):
        # Send the data and label to the device
        data, original_target = original_image.to(device), original_target.to(device)
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        # Forward pass the data through the model
        output = self.model(original_image)
        init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        original_confidence = output.max(1, keepdim=True)[0]

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != original_target.item():
            raise AttackError("Original image is already misclassified by model.")
        # Calculate the loss
        loss = F.nll_loss(output, original_target)
        # Zero all existing gradients
        self.model.zero_grad()
        # Calculate gradients of model in backward pass
        loss.backward()
        # Collect datagrad
        data_grad = data.grad.data
        # Call FGSM Attack
        adversarial_image, gradient_image = self.calc_adversarial(data, epsilon, data_grad)
         # Re-classify the perturbed image
        output = self.model(adversarial_image)
        adversarial_target = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        adversarial_confidence = output.max(1, keepdim=True)[0]
        if adversarial_target == original_target:
            raise AdversarialNotFoundError
        attack_result = AttackResult(original_image=original_image.detach(),
                     original_target=original_target,
                     original_confidence=original_confidence,
                     adversarial_image=adversarial_image.detach(),
                     adversarial_target=adversarial_target,
                     adversarial_confidence=adversarial_confidence,
                     gradient_image=gradient_image)

        return attack_result

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: " + str(device))

    # instantiate model
    loss_function = nn.NLLLoss()
    model = ModelRepository.get_model(FFModel, MNISTDataset)

    # load data set
    feature_transform_fun = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_path = os.path.join(datasets_path, "mnist")
    dataset = MNISTDataset.load(mnist_path, feature_transform_fun=feature_transform_fun)
    dataset_loader = DatasetLoader(dataset)

    fgsm = FGSM(model=model)
    iterator = iter(dataset_loader)
    original_image, _, original_target = next(iterator)
    attack_result = fgsm.search_for_adversarial_example(original_image, original_target, epsilon=100)
    attack_result.visualize()





