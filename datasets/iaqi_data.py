#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-24-20 22:38
# @Author  : Kan Huang (kan.huang@connect.ust.hk)

"""IAQI data.
Individual Air Quality Index (IAQI) data loader.
"""
import numpy as np
import pandas as pd
from datasets.window_sequences import WindowSequences, MultiChannelWindowSequences


def load_data(data_name, window_size, stride, batch_size=32, validation_split=0, shuffle=True, seed=42, standard="USA"):
    """ Load pm25 data
    # Arguments
        data_name: data name, e.g., "pm25_0", "pm25_all".
        window_size: window size of the window sequence.
        stride: stride of the window sequence sampler.
        batch_size: batch_size of the window sequence data generator.
    """
    date = "20191125-2028"
    data_type = data_name[:4]
    all_channels = True if data_name[-3:] == "all" else False
    df_pm_iaqi = pd.read_pickle(
        f"./data/{date}/{data_type}_iaqi_{standard}.pkl")
    df_pm_level_polygonal = pd.read_pickle(
        f"./data/{date}/{data_type}_level_polygonal_{standard}.pkl")

    if all_channels:
        multi_sequences, multi_ys = [], []
        for i in range(4):
            data_name = data_type + f"_{i}"
            sequence = df_pm_iaqi[data_name].to_numpy()
            y = df_pm_level_polygonal[data_name].to_numpy()
            multi_sequences.append(sequence)
            multi_ys.append(y)
        window_sequence_train = MultiChannelWindowSequences(
            multi_sequences, multi_ys, window_size=window_size, stride=stride, batch_size=batch_size, shuffle=shuffle, seed=seed, validation_split=validation_split, subset="training")
        window_sequence_val = None if validation_split == 0 else MultiChannelWindowSequences(
            multi_sequences, multi_ys, window_size=window_size, stride=stride, batch_size=batch_size, shuffle=shuffle, seed=seed, validation_split=validation_split, subset="validation")
        num_samples = len(window_sequence_train.multi_channel_index)
    else:
        sequence = df_pm_iaqi[data_name].to_numpy()
        y = df_pm_level_polygonal[data_name].to_numpy()

        # train and val subset should use the same seed and validation_split
        window_sequence_train = WindowSequences(
            sequence, y, window_size=window_size, stride=stride, batch_size=batch_size, shuffle=shuffle, seed=seed, validation_split=validation_split, subset="training")
        window_sequence_val = None if validation_split == 0 else WindowSequences(
            sequence, y, window_size=window_size, stride=stride, batch_size=batch_size, shuffle=shuffle, seed=seed, validation_split=validation_split, subset="validation")
        num_samples = len(window_sequence_train.index)

    num_batches = len(window_sequence_train)

    print("Training subset:")
    print(f"num_samples: {num_samples}")
    print(f"num_batches: {num_batches}")
    print(
        f"Total window data size: {window_size*num_samples}")

    return window_sequence_train, window_sequence_val


def main():
    pass


if __name__ == "__main__":
    main()
