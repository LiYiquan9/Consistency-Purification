import bisect 
import os
import pickle

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.datasets.utils import check_integrity
from typing import *
from zipdata import ZipData

IMAGENET_DIR = "/home/datasets/imagenet"

# list of all datasets
DATASETS = ["imagenet", "imagenet32", "cifar10"]


def get_dataset(dataset: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "cifar10":
        return _cifar10(split)
    


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar10":
        return 10
    elif dataset == 'lsun10':
        return 10


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "imagenet32":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)


def get_input_center_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's Input Centering layer"""
    if dataset == "imagenet":
        return InputCenterLayer(_IMAGENET_MEAN)
    elif dataset == "cifar10":
        return InputCenterLayer(_CIFAR10_MEAN)


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]


def _cifar10(split: str) -> Dataset:
    dataset_path = os.path.join('datasets', 'dataset_cache')
    if split == "train":
        # return datasets.CIFAR10(dataset_path, train=True, download=True, transform=transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor()
        # ]))
        return datasets.CIFAR10(dataset_path, train=True, download=True, transform=transforms.ToTensor())
    elif split == "test":
        return datasets.CIFAR10(dataset_path, train=False, download=True, transform=transforms.ToTensor())

    else:
        raise Exception("Unknown split name.")
    


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means)/sds


class InputCenterLayer(torch.nn.Module):
    """Centers the channels of a batch of images by subtracting the dataset mean.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(InputCenterLayer, self).__init__()
        self.means = torch.tensor(means).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return input - means


class TiTop50KDataset(Dataset):
            """500K images closest to the CIFAR-10 dataset from 
               the 80 Millon Tiny Images Datasets"""
            def __init__(self):
                super(TiTop50KDataset, self).__init__()
                dataset_path = os.path.join('datasets', 'ti_top_50000_pred_v3.1.pickle')

                self.dataset_dict = pickle.load(open(dataset_path,'rb'))
                #{'data', 'extrapolated_targets', 'ti_index', 
                # 'prediction_model', 'prediction_model_epoch'}
                
                self.length = len(self.dataset_dict['data'])
                self.transforms = transforms.Compose([
                            transforms.Resize((32,32)),
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()
                        ])

            def __getitem__(self, index):
                img = self.dataset_dict['data'][index]
                target = self.dataset_dict['extrapolated_targets'][index]
                
                img = Image.fromarray(img)
                img = self.transforms(img)
        
                return img, target
                        
            def __len__(self):
                return self.length


class MultiDatasetsDataLoader(object):
    """Dataloader to alternate between batches from multiple dataloaders 
    """
    def __init__(self, task_data_loaders, equal_num_batch=True, start_iteration=0):
        if equal_num_batch:
            lengths = [len(task_data_loaders[0]) for i,_ in enumerate(task_data_loaders)]
        else:
            lengths = [len(data_loader) for data_loader in task_data_loaders]
    
        self.task_data_loaders = task_data_loaders
        self.start_iteration = start_iteration
        self.length = sum(lengths)
        self.dataloader_indices = np.hstack([
            np.full(task_length, loader_id)
            for loader_id, task_length in enumerate(lengths)
        ])

    def __iter__(self):
        self.task_data_iters = [iter(data_loader)
                                for data_loader in self.task_data_loaders]
        self.cur_idx = self.start_iteration
        # synchronizing the task sequence on each of the worker processes
        # for distributed training. The data will still be different, but
        # will come from the same task on each GPU.
        # np.random.seed(22)
        np.random.shuffle(self.dataloader_indices)
        # np.random.seed()
        return self

    def __next__(self):
        if self.cur_idx == len(self.dataloader_indices):
            raise StopIteration
        loader_id = self.dataloader_indices[self.cur_idx]
        self.cur_idx += 1
        return next(self.task_data_iters[loader_id]), loader_id

    next = __next__  # Python 2 compatibility

    def __len__(self):
        return self.length

    @property
    def num_tasks(self):
        return len(self.task_data_iters)
