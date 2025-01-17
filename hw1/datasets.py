import numpy as np

import torch
from torch.utils.data import Dataset


class RandomImageDataset(Dataset):
    """
    A dataset returning random noise images of specified dimensions
    """

    def __init__(self, num_samples, num_classes, C, W, H):
        """
        :param num_samples: Number of samples (labeled images in the dataset)
        :param num_classes: Number of classes (labels)
        :param C: Number of channels per image
        :param W: Image width
        :param H: Image height
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.image_dim = (C, W, H)

    def __getitem__(self, index):
        """
        Returns a labeled sample.
        :param index: Sample index
        :return: A tuple (sample, label).
        """
        Dataset
        # TODO:
        #  Create a random image tensor and return it.
        #  Try to make sure to always return the same image for the
        #  same index (make it deterministic per index), but don't mess-up
        #  the random state outside this method.

        # For Torch to generate the same numbers for "rand(x1)", "rand(x2)"
        # where x1=x2 we need to fix the seed. this can be done by "manual_seed"
        torch.manual_seed(index)

        # now we need to create a random label and a random image
        # the label needs to be within the range of 0 and the number of classes
        label = torch.randint(low=0, high=(self.num_classes-1), size=(1,), dtype=torch.int)

        # the image needs to be within a certain range of grey
        grey_range = 255
        image = torch.rand(*self.image_dim, dtype=torch.float) * grey_range

        return image, label.item()

    def __len__(self):
        """
        :return: Number of samples in this dataset.
        """
        return self.num_samples


class SubsetDataset(Dataset):
    """
    A dataset that wraps another dataset, returning a subset from it.
    """
    def __init__(self, source_dataset: Dataset, subset_len, offset=0):
        """
        Create a SubsetDataset from another dataset.
        :param source_dataset: The dataset to take samples from.
        :param subset_len: The total number of sample in the subset.
        :param offset: The offset index to start taking samples from.
        """
        if offset + subset_len > len(source_dataset):
            raise ValueError("Not enough samples in source dataset")

        self.source_dataset = source_dataset
        self.subset_len = subset_len
        self.offset = offset

    def __getitem__(self, index):
        # TODO:
        #  Return the item at index + offset from the source dataset.
        #  Raise an IndexError if index is out of bounds.

        real_index = self.offset + index
        if index >= self.subset_len:
            raise IndexError("Not enough samples in source dataset")

        return self.source_dataset[real_index]
        #return self.source_dataset.__getitem__(index=real_index)

    def __len__(self):
        return self.subset_len


