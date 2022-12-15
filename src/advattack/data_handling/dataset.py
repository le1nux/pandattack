#!/usr/bin/env python3

import torch.utils.data.dataset as torch_dataset
import os
from abc import abstractmethod
from typing import List, Dict
from advattack.error_handling.exception import DatasetNotFoundError, DatasetFileCorruptError
import shutil
from torchvision.datasets.utils import download_url
import gzip
from advattack.util.logger import logger
import random
from typing import TypeVar, Type
import hashlib
import tarfile

T = TypeVar('T', bound='Dataset')


class Dataset(torch_dataset.Dataset):
    """Dataset class inherited from torch's Dataset class that adds further functionality.

    Every other Dataset representation has to inherit from or instantiate this class.
    """
    # TODO: Fix this ugly hard coding ...
    logger = logger.getChild("data_handling.dataset.Dataset")

    def __init__(self, root_path: str, feature_transform_fun=None, target_transform_fun=None):
        """
        :param root_path: Path to dataset
        :param feature_transform_fun: Function performing transformation on the dataset's features
        :param target_transform_fun: Function performing transformation on the dataset's targets
        :param force_download: If `True`, the dataset will be downloaded again and the prior one on disk will be replaced
        """
        self.root_path = os.path.expanduser(root_path)
        self.feature_transform_fun = feature_transform_fun
        self.target_transform_fun = target_transform_fun

        if not self.check_exists(self.root_path):
            raise DatasetNotFoundError(f'Dataset not found in {self.root_path}. You can use Dataset.create_dataset(...) to download it')
        self.samples, self.labels, self.label_map = self.load_data_from_disc()

    @classmethod
    def get_dataset_identifier(cls):
        return cls.__mro__[0].__name__

    @abstractmethod
    def visualize_samples(self, min_index: int, max_index: int):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls: Type[T], path: str, feature_transform_fun=None, target_transform_fun=None) -> T:
        raise NotImplementedError

    @staticmethod
    def extract_gzip(gzip_path: str, remove_finished=False):
        destination_path = gzip_path.replace('.gz', '')
        print('Extracting {}'.format(gzip_path))
        with open(destination_path, 'wb') as out_f, \
                gzip.GzipFile(gzip_path) as zip_f:
            out_f.write(zip_f.read())
        if remove_finished:
            os.unlink(gzip_path)
        return destination_path

    @staticmethod
    def extract_tar(archive_path: str, remove_finished=False):
        if archive_path.endswith(".tar.gz"):
            with tarfile.open(archive_path, 'r:gz') as tar:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, os.path.dirname(archive_path))
        elif archive_path.endswith(".tar"):
            with tarfile.open(archive_path, 'r') as tar:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, os.path.dirname(archive_path))
        if remove_finished:
            os.unlink(archive_path)

    @classmethod
    @abstractmethod
    def check_exists(cls, root_path: str) -> bool:
        raise NotImplementedError

    ###########################  DATASET DOWNLOAD AND EXTRACTION FUNCTIONS ##################################

    @classmethod
    @abstractmethod
    def create_dataset(cls, root_path: str) -> str:
        raise NotImplementedError


    @classmethod
    def populate_data_repository(cls, root_path: str, dataset_sources:Dict, force_download=False):
        if force_download or not cls.check_exists(root_path):
            # delete old repository if it exists
            cls.empty_repository(root_path=root_path)
            dataset_dirs = dataset_sources.copy()
            for source, (md5, directory, extraction_func) in dataset_dirs.items():
                # download and extract files
                download_directory = os.path.join(root_path, directory)
                downloaded_file_path = cls.download_dataset_file(url=source, path=download_directory, md5=md5)
                cls.extract_dataset_file(source_path=downloaded_file_path, extraction_fun=extraction_func)
        else:
            cls.logger.warn(f"Repository already exists in {root_path} and donwload not enforced.")


    @classmethod
    def empty_repository(cls, root_path: str):
        cls.logger.info(f"Deleting current repository {root_path} ...")
        # remove possibly existing dataset
        shutil.rmtree(root_path, ignore_errors=True)
        # create dataset directory
        os.makedirs(root_path)

    @classmethod
    def download_dataset_file(cls, path: str, url: str, md5: str) -> str:
        """ Downloads the dataset given by the url.
        :param path: path to the root of the data repository
        :param url: URL to the dataset
        :return: Path to downloaded file
        """
        cls.logger.info(f"Downloading data file from {url} ...")
        # download file
        filename = url.rpartition('/')[2]
        download_url(url, root=path, filename=filename)
        file_path = os.path.join(path, filename)
        if not Dataset.check_md5(file_path=file_path, md5=md5):
            cls.logger.warn(f"Given MD5 hash did not match with the md5 has of file {file_path}")
            raise DatasetFileCorruptError
        else:
            cls.logger.debug(f"Hash of file {file_path} is correct.")

        cls.logger.info("Done.")
        return file_path

    @staticmethod
    def calculate_md5(file_path: str, chunk_size: int = 1024 * 1024):
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                md5.update(chunk)
        return md5.hexdigest()

    @staticmethod
    def check_md5(file_path: str, md5: str):
        if not os.path.isfile(file_path):
            Dataset.logger.warn(f"File {file_path} not found!")
            return False
        file_md5 = Dataset.calculate_md5(file_path)
        return file_md5 == md5

    @classmethod
    def extract_dataset_file(cls, source_path: str, extraction_fun):
        """ The extraction function passed as a function argument must only take the dataset file path as input.
        All extraction parameters therefore already have to be present within the extraction function.
        A good approach is to wrap the respective extraction function implemented in this class
        using e.g. a lambda function.

        :param file_path:
        :param extraction_fun:
        :return:
        """
        cls.logger.info(f"Extracting {source_path} ...")
        extraction_fun(source_path)
        cls.logger.info("Done.")


    #######################################################################################################

    @abstractmethod
    def load_data_from_disc(self) -> (List, List, List):
        """ Method that implements loading functionality of an on disk dataset.

        :param folder_path: Path to dataset folder
        :param data_files:
        :return: samples, targets
        """
        raise NotImplementedError()

    def feature_transform(self, sample):
        if self.feature_transform_fun is not None:
            return self.feature_transform_fun(sample)
        else:
            return sample

    def target_transform(self, target):
        if self.target_transform_fun is not None:
            return self.target_transform_fun(target)
        else:
            return target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        """ Returns the sample and target of the dataset at given index position.
        :param index: index within dataset
        :return: sample, target
        """
        return self.feature_transform(self.samples[index]), self.target_transform(self.labels[index])

    def get_train_and_validation_set_indices(self, train_valid_split_ratio=0.8, seed=1):
        random.seed(seed)
        indices = list(range(len(self)))
        random.shuffle(indices)
        split_index = int(train_valid_split_ratio * len(indices))
        train_indices, valid_indices = indices[:split_index], indices[split_index:]
        return train_indices, valid_indices
