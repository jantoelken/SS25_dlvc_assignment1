import pickle
from typing import Tuple
import numpy as np
import os

import torch

from assignment_1_code.datasets.dataset import Subset, ClassificationDataset


class CIFAR10Dataset(ClassificationDataset):
    """
    Custom CIFAR-10 Dataset.
    """

    def __init__(self, fdir: str, subset: Subset, transform=None):
        """
        Initializes the CIFAR-10 dataset.
        """
        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

        self.fdir = fdir
        self.subset = subset
        self.transform = transform

        self.images, self.labels = self.load_cifar()

    def load_cifar(self) -> Tuple:
        """
        Loads the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all images from "data_batch_5".
          - The test set contains all images from "test_batch".

        Depending on which subset is selected, the corresponding images and labels are returned.

        Images are loaded in the order they appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in RGB channel order.
        """

        if not os.path.isdir(self.fdir):
            raise ValueError(f"{self.fdir} is not a directory")
        if self.subset == Subset.TRAINING:
            fnames = {"data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4"}
        elif self.subset == Subset.VALIDATION:
            fnames = {"data_batch_5"}
        elif self.subset == Subset.TEST:
            fnames = {"test_batch"}
        else:
            raise ValueError(f"Unknown subset {self.subset}")
        images = []
        labels_local = []
        for fname in fnames:
            fpath = os.path.join(self.fdir, fname)
            if not os.path.isfile(fpath):
                raise ValueError(f"File {fpath} is missing")
            with open(fpath, "rb") as fo:
                batch = pickle.load(fo, encoding="bytes")
                data = batch[b"data"]
                labels = batch[b"labels"]
                # reshape to (10000, 3, 32, 32) -> transpose to (10000, 32, 32, 3)
                data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                images.append(data)
                labels_local.extend(labels)
        return np.concatenate(images), np.array(labels_local)

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns the idx-th sample in the dataset, which is a tuple,
        consisting of the image and labels.
        Applies transforms if not None.
        Raises IndexError if the index is out of bounds.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds")
        
        image = torch.from_numpy(self.images[idx]).permute(2,0,1)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)
        return image, label

    def num_classes(self) -> int:
        """
        Returns the number of classes.
        """
        return len(self.classes)
