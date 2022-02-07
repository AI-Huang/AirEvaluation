#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Oct-12-20 16:45
# @Author  : Kan Huang (kan.huang@connect.ust.hk)

"""window_sequence_test
Tests window_sequence module.
"""
import numpy as np
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.platform import test
from window_sequences import WindowSequences, MultiChannelWindowSequences


def random_sequence(N):
    """generate a random sequence for testing
    Inputs:
        N: length of the sequence,
        window_size: window size when sampling the sequence,
        stride: stride when taking windows from the sequence.
    """

    sequence = np.random.rand(N)
    y = np.random.rand(N)

    return sequence, y


class TestWindowSequences(keras_parameterized.TestCase):

    def test_get_batches(self):
        """test_get_batches,
        test stride and number of samples,
        test accessing batches
        """
        # print(self.test_get_batches.__doc__)
        N, window_size, stride = 219989, 7200, 10

        sequence, y = random_sequence(N=N)

        batch_size = 32
        shuffle = True
        seed = 42
        validation_split = 0.2

        window_sequences_train = WindowSequences(
            sequence, y, window_size=window_size, stride=stride, batch_size=batch_size, shuffle=shuffle, seed=seed, validation_split=validation_split, subset="training")
        window_sequence_val = WindowSequences(
            sequence, y, window_size=window_size, stride=stride, batch_size=batch_size, shuffle=shuffle, seed=seed, validation_split=validation_split, subset="validation")

        print(f"num train samples: {len(window_sequences_train.index)}")
        print(f"num train batches: {len(window_sequences_train)}")

        print("Test getting batches...")
        from tqdm import tqdm
        for i in tqdm(range(len(window_sequences_train))):
            batch_x, batch_y = window_sequences_train.__getitem__(i)

    def test_split(self):
        """test_split, train val split test
        """
        N, window_size, stride = 219989, 7200, 10
        sequence, y = random_sequence(N=N)
        validation_split = 0.2
        shuffle = True
        seed = 42

        window_sequence_all = WindowSequences(
            sequence, y, window_size=window_size, stride=stride, batch_size=32, shuffle=shuffle, seed=seed, validation_split=0, subset="training")

        # train and val subset should use the same seed and validation_split
        window_sequences_train = WindowSequences(
            sequence, y, window_size=window_size, stride=stride, batch_size=32, shuffle=shuffle, seed=seed, validation_split=validation_split, subset="training")
        window_sequence_val = WindowSequences(
            sequence, y, window_size=window_size, stride=stride, batch_size=32, shuffle=shuffle, seed=seed, validation_split=validation_split, subset="validation")

        index = np.concatenate([window_sequences_train.index,
                                window_sequence_val.index])
        """ the index will still distribute linearly
        >>> index = np.sort(index)
        >>> print(index)
        """
        """
        >>> import matplotlib.pyplot as plt
        >>> plt.scatter([_ for _ in range(len(index))], index)
        >>> plt.show()
        """
        """test number of samples of each subset
        """
        num_all = len(window_sequence_all.index)
        num_train = len(window_sequences_train.index)
        num_val = len(window_sequence_val.index)

        print(f"validation_split: {validation_split}")
        print(f"num_all: {num_all}")
        print(f"num_train: {num_train}")
        print(f"num_val: {num_val}")

    def test_on_epoch_end(self):
        N, window_size, stride = 219989, 7200, 10
        sequence, y = random_sequence(N=N)
        seed = 42
        shuffle = True
        validation_split = 0.2
        window_sequences_train = WindowSequences(
            sequence, y, window_size=window_size, stride=stride, batch_size=32, shuffle=shuffle, seed=seed, validation_split=validation_split, subset="training")

        print(len(window_sequences_train.index))
        print(len(window_sequences_train))

        """
        # index won't change, but their order changes.
        for _ in range(10):
            window_sequences_train.on_epoch_end()
            num_train = len(window_sequences_train.index)
            print(f"num_train: {num_train}")
            # print(window_sequences_train.index)
            print(np.sum(window_sequences_train.index))
        """


class TestMultiChannelWindowSequences(keras_parameterized.TestCase):

    def test_get_batches(self):
        N, window_size, stride = 219989, 7200, 10
        channels = 4

        sequence, y = random_sequence(N=N)
        multi_sequences = [sequence for _ in range(channels)]
        multi_ys = [y for _ in range(channels)]

        batch_size = 32
        shuffle = True
        seed = 42
        validation_split = 0.2

        window_sequences_train = MultiChannelWindowSequences(
            multi_sequences, multi_ys, window_size=window_size, stride=stride, batch_size=batch_size, shuffle=shuffle, seed=seed, validation_split=validation_split, subset="training")
        window_sequence_val = MultiChannelWindowSequences(
            multi_sequences, multi_ys, window_size=window_size, stride=stride, batch_size=batch_size, shuffle=shuffle, seed=seed, validation_split=validation_split, subset="validation")

        print(
            f"Train samples: {len(window_sequences_train.multi_channel_index)}.")
        print(f"Val samples: {len(window_sequence_val.multi_channel_index)}.")

        print("Test getting batches...")
        from tqdm import tqdm
        for i in tqdm(range(len(window_sequences_train))):
            batch_x, batch_y = window_sequences_train.__getitem__(i)

    def test_split(self):
        pass


if __name__ == "__main__":
    test.main()
