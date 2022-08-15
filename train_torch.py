#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-13-20 19:24
# @Author  : Kan Huang (kan.huang@connect.ust.hk)

import os
import csv
import pandas as pd
import torch
from torch import nn
import torch.utils.data as data
from wavenet.torch_fn.window_sequence import WindowSequence, set_subset
from wavenet.torch_fn.wavenet_lstm import WaveNet_LSTM


def train(model, dataset, epochs=50, batch_size=32, use_cuda=False, seed=None):
    """training process, use a model to train data
    Inputs:
        model: model.
        dataset: dataset to be trained.
        epochs: training epochs.
        seed:
    """
    num_train = len(dataset)

    if seed:  # set RNG seed
        torch.manual_seed(seed)

    model.train()

    criterion = nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, betas=(0.9, 0.99))

    headers = ["epoch", 'loss']
    f = open("./logs/log_torch_mnist.csv", 'w', newline='')
    f_csv = csv.writer(f)
    f_csv.writerow(headers)

    for epoch in range(epochs):  # a total iteration/epoch
        for batch_idx, (X_batch, y_batch) in enumerate(dataset):
            X_batch = torch.from_numpy(X_batch).float()
            y_batch = torch.from_numpy(y_batch).float()
            if use_cuda:  # config device
                X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:  # print every 100 steps
                print(
                    f"Train epoch: {epoch}, [{batch_idx*batch_size}/{num_train} ({batch_idx*batch_size/num_train*100:.2f}%)].\tLoss: {loss:.6f}")
                f_csv.writerow([epoch, float(loss)])


def main():
    # Load pm25 data
    date = "20191125-2028"
    standard = "USA"
    prefix = os.path.join("~", ".datasets", "AirMonitor", "data", f"{date}")
    df_pm25_iaqi = pd.read_pickle(
        os.path.join(prefix, f"pm25_iaqi_{standard}.pkl")
    )
    df_pm25_level_polygonal = pd.read_pickle(
        os.path.join(prefix, f"pm25_level_polygonal_{standard}.pkl")
    )

    data_name = "pm25_0"  # data name, e.g., "pm25_0"
    sequence = df_pm25_iaqi[data_name].to_numpy()
    y = df_pm25_level_polygonal[data_name].to_numpy()

    dataset = WindowSequence(
        sequence, y, window_size=7200, stride=10, shuffle=True)
    train_subset, val_subset = set_subset(dataset, validation_split=0.2)

    # Create model
    model = WaveNet_LSTM(input_size=7200)

    from torch.utils.tensorboard import SummaryWriter
    # default `log_dir` is "runs" - we'll be more specific here
    # writer = SummaryWriter('runs/debug')
    # writer.add_graph(model)
    # writer.close()

    # Train model
    use_cuda = torch.cuda.is_available()  # CUDA
    model = model.cuda() if use_cuda else model

    train(model, train_subset, epochs=1, batch_size=32, use_cuda=use_cuda)


if __name__ == "__main__":
    main()
