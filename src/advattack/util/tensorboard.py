import os
import shutil
from advattack import tensorboard_path
from torch.utils.tensorboard import SummaryWriter
from enum import Enum


class TensorboardMode(Enum):
    TRAIN = "train"
    ATTACK = "attack"
    DEFENSE = "defense"


class TensorboardWrapper:

    @staticmethod
    def get_summary_writer(model_identifier, dataset_identifier, mode: TensorboardMode):
        # set up tensorboard
        tb_directory = os.path.join(tensorboard_path, str(mode.value), dataset_identifier, model_identifier)
        # delete if exists and create new folder
        shutil.rmtree(tb_directory, ignore_errors=True)
        os.makedirs(os.path.dirname(tb_directory), exist_ok=True)

        tb_writer = SummaryWriter(log_dir=tb_directory, flush_secs=10)
        return tb_writer
