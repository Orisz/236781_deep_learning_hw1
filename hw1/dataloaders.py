import math

import numpy as np
import torch
import torch.utils.data.sampler as sampler
from torch.utils.data import Dataset


def create_train_validation_loaders(dataset: Dataset, validation_ratio,
                                    batch_size=100, num_workers=2):
    """
    Splits a dataset into a train and validation set, returning a
    DataLoader for each.
    :param dataset: The dataset to split.
    :param validation_ratio: Ratio (in range 0,1) of the validation set size to
        total dataset size.
    :param batch_size: Batch size the loaders will return from each set.
    :param num_workers: Number of workers to pass to dataloader init.
    :return: A tuple of train and validation DataLoader instances.
    """
    if not(0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)

    # TODO:
    #  Create two DataLoader instances, dl_train and dl_valid.
    #  They should together represent a train/validation split of the given
    #  dataset. Make sure that:
    #  1. Validation set size is validation_ratio * total number of samples.
    #  2. No sample is in both datasets. You can select samples at random
    #     from the dataset.
    #  Hint: you can specify a Sampler class for the `DataLoader` instance
    #  you create.
    # ====== YOUR CODE: ======
    indices = np.random.permutation(len(dataset))
    
    last_valid_index = int(validation_ratio * len(dataset))
    valid_indices, train_indices = indices[:last_valid_index], indices[last_valid_index:]
#     train_len = len(dataset) - last_valid_index
#     valid_indices = list(range(last_valid_index))
#     train_indices = [i + last_valid_index for i in range(train_len)]
    

    train_sampler = sampler.SubsetRandomSampler(train_indices)
    valid_sampler = sampler.SubsetRandomSampler(valid_indices)

    dl_train = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers,
                                           sampler=train_sampler)
    dl_valid = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers,
                                           sampler=valid_sampler)
    # ========================

    return dl_train, dl_valid

