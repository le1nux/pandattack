from advattack import models_path
import os
from advattack.models.nn.net import NNModel
import json
import torch


class ModelRepository:

    @staticmethod
    def get_model(model_class, dataset_class, device=None) -> NNModel:
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_folder = os.path.join(models_path, dataset_class.get_dataset_type(), model_class.get_model_type())
        config_path = os.path.join(model_folder, "config.json")

        with open(config_path, "r") as f:
            model_config = json.load(f)
        model = model_class(**model_config).to(device)
        model_path = os.path.join(models_path, dataset_class.get_dataset_type(), model.get_model_type())
        model.load_model(model_path)
        return model

    @staticmethod
    def store_model(model: NNModel, dataset_class) -> str:
        model_folder = os.path.join(models_path, dataset_class.get_dataset_type(), model.get_model_type())
        config_path = os.path.join(model_folder, "config.json")
        model.save_model(folder_path=model_folder)
        with open(config_path, "w") as f:
            config = model.get_config()
            json.dump(config, f)
        return model_folder
