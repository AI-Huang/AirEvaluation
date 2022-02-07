#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Aug-03-20 00:27
# @Update  : Oct-27-20 18:35
# @Update  : Dec-27-20 16:56
# @Author  : Kelley Kan Huang (kan.huang@connect.ust.hk)
# @RefLink : https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/utils/data_utils.py#L437


import numpy as np
import tensorflow as tf


class WindowSequences(tf.keras.utils.Sequence):
    def __init__(
        self,
        sequence,
        y,
        window_size,
        stride=1,
        batch_size=32,
        shuffle=False,
        seed=42,
        subset="training",
        # set to 0.2, if you want 0.8 of the samples as training set and 0.2 as validation set
        validation_split=0
    ):
        """Initialize the windowed sub-sequences dataset generated from a sequence data.

        Arguments:
            sequence: sequence data, a numpy array.
            y: corresponding y to the sub-sequences data.
            batch_size: batch size, default 32.
            shuffle: whether to shuffle the data index initially **and** after each epoch.
            seed: set the initial random seed for the first time of shuffling sample indices when using numpy.random.shuffle().
            subset: one of "training" and "validation", default "training". If the data is split into training and validation sets, the set this data generator belongs to.
            validation_split: ratio of validation set, e.g., set to 0.2, if you want 0.8 of the samples as training set and 0.2 as validation set.
        """
        self.sequence = sequence
        self.sequence_len = len(sequence)
        self.y = y
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size

        self.index = self._set_index_array()

        self.shuffle = shuffle

        if subset not in ["training", "validation"]:
            raise ValueError(
                """subset must be one of ["training", "validation"]!""")
        self.subset = subset

        # shuffle firstly, if shuffle
        if self.shuffle:
            # set random seet for the first time of shuffling
            np.random.seed(seed)
            self.shuffle_index()

        # split secondly
        self.validation_split = validation_split
        self.set_subset()

    def __getitem__(self, batch_index):
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

        sample_indices = \
            self.index[batch_index * batch_size:(batch_index+1) * batch_size]

        batch_x = np.empty((batch_size, window_size))
        batch_y = np.empty(batch_size)
        for _, i in enumerate(sample_indices):
            batch_x[_, ] = sequence[i:i + window_size]
            # label element on the right edge of the window
            batch_y[_] = y[i+window_size-1]  # must minus 1!
        # batch_x must have dimensions (N, D1, D2)
        batch_x = np.expand_dims(batch_x, -1)

        return batch_x, batch_y

    def __len__(self):
        """Number of batches in the Sequence.
        Returns:
            The number of batches in the Sequence.
        """
        return int(np.ceil(self.index.shape[0] / self.batch_size))

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        # shuffle data index on the end of every epoch if self.shuffle is True
        if self.shuffle:
            self.shuffle_index()

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

    def set_subset(self):
        """set_subset
        split indices according to the 'validation_split' and 'subset' attribute,
        this code will change the 'index' attribute.
        """
        if 0 <= self.validation_split < 1:
            i = int(self.validation_split * len(self.index))
            # cut ending elements from the ith index
            if self.subset == "training":
                # the ending part is the training set
                self.index = self.index[i:]
            elif self.subset == "validation":
                # the front part is the validation set
                self.index = self.index[:i]

    def shuffle_index(self):
        """shuffle data index
        """
        np.random.shuffle(self.index)

    def normalize_y(self):
        """Normalize y on the whole dataset
        """
        self.y_mean = self.y.mean()
        self.y_std = self.y.std()
        self.y = (self.y - self.y_mean) / self.y_std

    def denormalize_y(self):
        raise NotImplementedError


class MultiChannelWindowSequences(WindowSequences):
    """Multi-channel window sequence data generator.
    # The data combination logic

    When this class initializes, its super class `WindowSequences` will initialize firstly. During the initialization of `WindowSequences`, `self.index` will be initialized, this is used as samples' index when indexing single channel window sequence, also, it is used as *base index* in `MultiChannelWindowSequences` class.

    However, only length (`len(self.index)`) and range (`max(self.index)`) of `self.index` are used, for `self.multi_channel_index` attribute of `MultiChannelWindowSequences` subclass to initialize and be generated.

    # The data shuffle logic
    """

    def __init__(
        self,
        multi_sequences: list,
        multi_ys: list,
        window_size,
        stride=1,
        batch_size=32,
        shuffle=True,
        seed=42,
        subset="training",
        # 0.2, if you want 0.8 of the samples as training set and 0.2 as validation set
        validation_split=0
    ):
        """Initialize the windowed sub-sequences dataset generated from multi sequences data.

        Arguments:
            multi_sequences: a list of sequences data, each element is a numpy array.
            multi_y: a list of corresponding y to the sub-sequences data.
            batch_size: batch size, default 32.
            shuffle: whether to shuffle the data index initially **and** after each epoch.
            seed: set the initial random seed for the first time of shuffling sample indices when using numpy.random.shuffle().
            subset: one of "training" and "validation", default "training". If the data is split into training and validation sets, the set this data generator belongs to.
            validation_split: ratio of validation set, e.g., set to 0.2, if you want 0.8 of the samples as training set and 0.2 as validation set.
        """
        super().__init__(multi_sequences[0],
                         multi_ys[0],
                         window_size,
                         stride=stride,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         seed=seed,
                         subset=subset,
                         # 0.2, if you want 0.8 of the samples as training set and 0.2 as validation set
                         validation_split=validation_split)
        self.n = len(multi_sequences)  # channels

        self.multi_sequences, self.multi_ys = self.process_multi_sequences(
            multi_sequences, multi_ys)

        self.multi_channel_index = self._set_multi_channel_index_array()

        # shuffle firstly, if shuffle
        if self.shuffle:
            # set random seet for the first time of shuffling
            np.random.seed(seed)
            self.shuffle_multi_index()

    def process_multi_sequences(self, multi_sequences, multi_ys):
        # Concatenate
        return np.concatenate(multi_sequences), np.concatenate(multi_ys)

    def _set_multi_channel_index_array(self):
        """_set_multi_channel_index_array
        after super()._set_index_array()
        _set_multi_channel_index_array()
        """
        len_base_index = len(self.index)
        # hash function: channel_no, base_index -> channel_masked index
        multi_channel_index = np.empty(len_base_index * self.n)
        for i in range(self.n):
            multi_channel_index[i * len_base_index: (i+1)*len_base_index] \
                = self.hash_index(self.index, i)
        return multi_channel_index

    def hash_index(self, indices, channel):
        """hash_index
        # Inputs:
            indices: numpy array or scalar
            channel: scalar
        """
        # hash function, (max(self.indices)+42) times 0, 1, 2, 3
        return indices + channel * (np.max(indices)+42)

    def dehash_index(self, hash_indices):
        """dehash_index
        # Inputs:
            index: array or scalar
            channel: scalar
        """
        # batch_size = len(hash_indices)
        # indices = np.empty(batch_size)
        # channels = np.empty(batch_size)
        # for _index in hash_indices:
        #     channel = int(_index // (max(self.index)+42))
        #     i = int(_index % (max(self.index)+42))
        # Vectorized
        channels = (hash_indices // (np.max(self.index)+42)).astype(int)
        indices = (hash_indices % (np.max(self.index)+42)).astype(int)

        return indices, channels

    def shuffle_multi_index(self):
        """shuffle multi data index
        """
        np.random.shuffle(self.multi_channel_index)

    def __len__(self):
        """Number of batches in the Sequence. Override super().__len__().
        Returns:
            The number of batches in the Sequence.
        """
        return super().__len__() * self.n

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        # shuffle data index on the end of every epoch if self.shuffle is True
        if self.shuffle:
            print("on_epoch_end(), shuffle_multi_index")
            self.shuffle_multi_index()

    def __getitem__(self, batch_index):
        """Gets batch at batch_index `batch_index`.

        Arguments:
            batch_index: batch_index of the batch in the Sequence.

        Returns:
            batch_x, batch_y: a batch of sequence data.
        """
        sequence = self.multi_sequences
        y = self.multi_ys

        window_size = self.window_size
        batch_size = self.batch_size

        sample_indices = \
            self.multi_channel_index[batch_index *
                                     batch_size:(batch_index+1) * batch_size]

        batch_x = np.empty((batch_size, window_size))
        batch_y = np.empty(batch_size)

        indices, channels = self.dehash_index(sample_indices)
        for _, i in enumerate(indices):
            channel = channels[_]

            offset = self.sequence_len * channel
            batch_x[_, ] = sequence[offset+i:
                                    offset+i + window_size]
            # label element on the right edge of the window
            batch_y[_] = y[offset + i +
                           window_size-1]  # must minus 1!
        # Vectorized
        # offsets =

        # batch_x must have dimensions (N, D1, D2)
        batch_x = np.expand_dims(batch_x, -1)

        return batch_x, batch_y
