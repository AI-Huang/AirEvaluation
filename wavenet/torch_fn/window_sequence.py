#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-12-20 22:56
# @Author  : Kelly Hwong (kan.huang@connect.ust.hk)
# @RefLink : https://github.com/pytorch/vision/blob/master/torchvision/datasets/vision.py
# @RefLink : https://pytorch.org/docs/stable/_modules/torchvision/datasets/mnist.html

import os
import numpy as np
import torch
import torch.utils.data as data
from typing import Any, Tuple


class WindowSequence(data.Dataset):
    """Initialize the windowed sub-sequences dataset generated from a sequence data.

    Args:
        sequence: sequence data, a numpy array.
        y: corresponding y to the sub-sequences data.
        batch_size: batch size, default 32.
        shuffle: whether to shuffle the data index initially **and** after each epoch.
        seed: set the initial random seed for the first time of shuffling sample indices when using numpy.random.shuffle().
        subset: one of "training" and "validation", default "training". If the data is split into training and validation sets, the set this data generator belongs to.
        validation_split: ratio of validation set, e.g., set to 0.2, if you want 0.8 of the samples as training set and 0.2 as validation set.
    """

    def __init__(
        self,
        sequence,
        y,
        window_size,
        stride=1,
        batch_size=32,
        shuffle=True,
        seed=42
    ):

        self.sequence = sequence
        self.y = y
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size

        self.indices = self._set_index_array()

        self.shuffle = shuffle

        if self.shuffle:
            # set random seet for the first time of shuffling
            np.random.seed(seed)
            self.shuffle_index()

    def __getitem__(self, batch_index: int) -> Tuple[Any, Any]:
        """Gets batch at batch_index `batch_index`.

        Arguments:
            batch_index: batch_index of the batch in the Sequence.

        Returns:
            batch_x, batch_y: a batch of sequence data.
        """
        sequence = self.sequence
        y = self.y
        window_size = self.window_size

        batch_size = self.batch_size
        sample_index = self.indices[batch_index *
                                    batch_size:(batch_index+1) * batch_size]

        batch_x = np.empty((batch_size, window_size))
        batch_y = np.empty((batch_size, 1))
        for _, i in enumerate(sample_index):
            batch_x[_, ] = sequence[i:i + window_size]
            # label element on the right edge of the window
            batch_y[_] = y[i+window_size-1]  # must minus 1!
        # batch_x must have dimensions (N, C_in, L_in) for PyTorch
        batch_x = np.expand_dims(batch_x, 1)
        return batch_x, batch_y

    def __len__(self) -> int:
        """Number of samples in the Dataset.
        Returns:
            The number of samples in the Dataset.
        """
        # return self.indices.shape[0] // self.batch_size + 1
        return len(self.indices)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item

    def _set_index_array(self):
        """_set_index_array
        """
        # index of the beginning element of the window samples
        # +1 makes the index space a close interval
        N = self.sequence.shape[0]
        window_size = self.window_size
        stride = self.stride
        return np.arange(0, N-window_size+1, step=stride)

    def shuffle_index(self):
        """shuffle data indices
        """
        np.random.shuffle(self.indices)

    def normalize_y(self):
        """Normalize y on the whole dataset
        """
        self.y_mean = self.y.mean()
        self.y_std = self.y.std()
        self.y = (self.y - self.y_mean) / self.y_std

    def denormalize_y(self):
        raise NotImplementedError


class WindowSequences(WindowSequence):
    pass


def set_subset(dataset, validation_split):
    """set_subset, split indices according to 'validation_split'.
    # Arguments:
        validation_split: 0.2, if you want 0.8 of the samples as training set and 0.2 as validation set.
    """
    assert 0 < validation_split < 1
    i = int(validation_split * len(dataset.indices))
    # cut ending elements from the ith index
    # the ending part is the training set
    train_subset = data.Subset(dataset=dataset, indices=dataset.indices[i:])
    # the front part is the validation set
    val_subset = data.Subset(dataset=dataset, indices=dataset.indices[:i])

    return train_subset, val_subset


def main():
    pass


if __name__ == "__main__":
    main()
