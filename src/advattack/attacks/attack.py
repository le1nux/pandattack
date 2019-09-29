from abc import abstractmethod
from advattack.models.nn.net import NNModel
import matplotlib.pyplot as plt
import numpy as np


class Attack:
    def __init__(self, model: NNModel, device):
        self.model = model
        self.device = device

    @abstractmethod
    def attack(self, original_image, original_target):
        raise NotImplementedError


class AttackResult:
    def __init__(self, original_image, original_target, original_confidence, adversarial_image, adversarial_target, adversarial_confidence, gradient_image):
        # TODO we need a custom reshape function here, that is passed as an argument to the function
        shape = np.sqrt(original_image[0].size()).astype(np.int32).repeat(2)
        self.original_image = original_image.view(*shape)
        self.original_target = int(original_target)
        self.original_confidence = np.e**float(original_confidence)
        self.adversarial_image = adversarial_image.view(*shape)
        self.adversarial_target = int(adversarial_target)
        self.adversarial_confidence = np.e**float(adversarial_confidence)
        self.gradient_image = gradient_image.view(*shape)

    def visualize(self):
        fig = plt.figure()
        fig.add_subplot(1, 3, 1)
        plt.imshow(self.original_image, cmap='gray')
        plt.title(f'Original\nTarget and prediction:{self.original_target}\nConf.:{round(self.original_confidence, 5)}')

        fig.add_subplot(1, 3, 2)
        plt.imshow(self.gradient_image, cmap='gray')
        plt.title(f'Gradient image\n\n')

        fig.add_subplot(1, 3, 3)
        plt.imshow(self.adversarial_image, cmap='gray')
        plt.title(f'Adversarial Example\nPrediction:{self.adversarial_target}\nConf.:{round(self.adversarial_confidence, 5)}')

        plt.show()
