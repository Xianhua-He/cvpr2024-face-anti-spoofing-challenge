import torch
import torch.utils.data
import torchvision
import logging
from operator import itemgetter
from torch.utils.data import DistributedSampler
from torch.utils.data import Dataset, Sampler
from datasets.CvprDataset_P1 import CvprDataset_P1
from datasets.CvprDataset_P21 import CvprDataset_P21
from datasets.CvprDataset_P22 import CvprDataset_P22

import numpy as np
from typing import Iterator, List, Optional, Union

class BalanceClassSampler(Sampler):
    """Abstraction over data sampler.
    Allows you to create stratified sample on unbalanced classes.
    """

    def __init__(self, dataset, mode: Union[str, int] = "upsampling"):
        """
        Args:
            labels (List[int]): list of class label
                for each elem in the dataset
            mode (str): Strategy to balance classes.
                Must be one of [downsampling, upsampling]
        """
        # super().__init__()

        self.indices = list(range(len(dataset)))
        labels = [self._get_label(dataset, idx) for idx in self.indices]

        labels = np.array(labels)
        samples_per_class = {
            label: (labels == label).sum() for label in set(labels)
        }
        logging.info('samples_per_class: %s', samples_per_class)

        self.lbl2idx = {
            label: np.arange(len(labels))[labels == label].tolist()
            for label in set(labels)
        }

        if isinstance(mode, str):
            assert mode in ["downsampling", "upsampling"]

        if isinstance(mode, int) or mode == "upsampling":
            samples_per_class = (
                mode
                if isinstance(mode, int)
                else max(samples_per_class.values())
            )
        else:
            samples_per_class = min(samples_per_class.values())

        self.labels = labels
        self.samples_per_class = samples_per_class
        self.length = self.samples_per_class * len(set(labels))

    def _get_label(self, dataset, idx):
        if isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        elif isinstance(dataset, CvprDataset_P1):
            return int(dataset.items[idx][1])
        elif isinstance(dataset, CvprDataset_P21):
            return int(dataset.items[idx][1])
        elif isinstance(dataset, CvprDataset_P22):
            return int(dataset.items[idx][1])
        else:
            raise NotImplementedError


    def __iter__(self) -> Iterator[int]:
        """
        Yields:
            indices of stratified sample
        """
        indices = []
        for key in sorted(self.lbl2idx):
            replace_flag = self.samples_per_class > len(self.lbl2idx[key])
            indices += np.random.choice(
                self.lbl2idx[key], self.samples_per_class, replace=replace_flag
            ).tolist()
        assert len(indices) == self.length
        np.random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """
        Returns:
             length of result sample
        """
        return self.length


class BalanceMultiClassSampler(Sampler):
    """Abstraction over data sampler.
    Allows you to create stratified sample on unbalanced classes.
    """

    def __init__(
        self, dataset, num_classes, mode: Union[str, int] = "upsampling"
    ):
        """
        Args:
            labels (List[int]): list of class label
                for each elem in the dataset
            mode (str): Strategy to balance classes.
                Must be one of [downsampling, upsampling]
        """
        # super().__init__()

        self.indices = list(range(len(dataset)))
        labels = [self._get_label(dataset, idx) for idx in self.indices]

        labels = np.array(labels)
        samples_per_class = {label_idx: 0 for label_idx in range(num_classes)}
        self.lbl2idx = {label_idx: [] for label_idx in range(num_classes)}

        for idx, label in enumerate(labels):
            for label_idx in range(num_classes):
                if label[label_idx] == 1:
                    samples_per_class[label_idx] += 1
                    self.lbl2idx[label_idx].append(idx)

        logging.info(samples_per_class)
        '''
        samples_per_class = {
            label: (labels == label).sum() for label in labels
        }

        self.lbl2idx = {
            label: np.arange(len(labels))[labels == label].tolist()
            for label in set(labels)
        }
        '''

        if isinstance(mode, str):
            assert mode in ["downsampling", "upsampling"]

        if isinstance(mode, int) or mode == "upsampling":
            samples_per_class = (
                mode
                if isinstance(mode, int)
                else max(samples_per_class.values())
            )
        else:
            samples_per_class = min(samples_per_class.values())

        self.labels = labels
        self.samples_per_class = samples_per_class
        self.length = self.samples_per_class * num_classes#len(set(labels))

    def _get_label(self, dataset, idx):
        if isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        else:
            raise NotImplementedError


    def __iter__(self) -> Iterator[int]:
        """
        Yields:
            indices of stratified sample
        """
        indices = []
        for key in sorted(self.lbl2idx):
            replace_flag = self.samples_per_class > len(self.lbl2idx[key])
            indices += np.random.choice(
                self.lbl2idx[key], self.samples_per_class, replace=replace_flag
            ).tolist()
        assert len(indices) == self.length
        np.random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """
        Returns:
             length of result sample
        """
        return self.length


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """
    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        elif self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class DatasetFromSampler(Dataset):
    """Dataset of indexes from `Sampler`."""
    def __init__(self, sampler: Sampler):
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        return len(self.sampler)

class DistributedSamplerWrapper(DistributedSampler):
    def __init__(self, sampler, num_replicas: Optional[int] = None, rank: Optional[int] = None, shuffle: bool = True,):
        super(DistributedSamplerWrapper, self).__init__(DatasetFromSampler(sampler), num_replicas=num_replicas, rank=rank, shuffle=shuffle,)
        self.sampler = sampler

    def __iter__(self):
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        return iter(itemgetter(*indexes_of_indexes)(self.dataset))

