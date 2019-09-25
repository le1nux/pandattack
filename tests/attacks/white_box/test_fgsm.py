#!/usr/bin/env python3
import pytest
from advattack.attacks.white_box.fgsm import FGSM
from advattack.models.nn.ff_net import FFModel
import torch


class TestFGSM:
    @pytest.fixture
    def model(self):
        model = FFModel(layers=[16, 5, 5])
        yield model

    @pytest.fixture
    def sample(self):
        yield torch.ones(16).view(1, -1)

    @pytest.fixture()
    def device(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        yield device

    def test_attack(self, sample, model, device):
        output = model.forward(sample)
        init_prediction = output.max(1, keepdim=True)[1][0]
        fgsm = FGSM(model=model, epsilon=100000, device=device)
        attack_result = fgsm.attack(sample, init_prediction)
        assert attack_result.adversarial_target != int(init_prediction)