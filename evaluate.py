#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Mar-06-21 20:40
# @Author  : Kan Huang (kan.huang@connect.ust.hk)


"""模型批量测试用代码
"""
import os
import argparse
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from datasets.window_sequences import WindowSequences
from datasets.data_utils import rel_time, iaqi
from datasets.iaqi_data import load_data
from utils.dir_utils import makedir_exist_ok
from utils.gpu_utils import get_gpu_memory, get_available_gpu_indices
from wavenet.keras_fn.wavenet_backup import WaveNet_LSTM, WaveNet_LSTM_ver2021, lr_schedule
import matplotlib.pyplot as plt

# tf.debugging.set_log_device_placement(True)

print("tensorflow version:", tf.__version__)


def evaluating_args():
    """parse arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_name', type=str, dest='data_name',
                        action='store', default=None, help='data_name, e.g., --data_name=pm25_0, data name from the sensor.')

    parser.add_argument('--stride', type=int, dest='stride',
                        action='store', default=10, help='stride, e.g., --stride=10, stride of the origin data to yield samples.')

    args = parser.parse_args()

    # 待测参数列表
    pm25_data_list = ["pm25_0", "pm25_1", "pm25_2", "pm25_3", "pm25_all"]
    stride_list = [10, 5, 2]

    # if data_name is not set
    if not args.data_name:
        args.data_name = "_".join([args.data_type, str(args.sensor_index)])

    if not args.data_name in pm25_data_list:
        raise ValueError(f"{args.data_name} NOT in pm25_data_list!")
    if not args.stride in stride_list:
        raise ValueError(f"{args.stride} NOT in stride_list!")

    return args


def main():
    # GPU config
    gpus_memory = get_gpu_memory()
    available_gpu_indices = get_available_gpu_indices(
        gpus_memory, required_memory=5000)
    model_gpu = available_gpu_indices[0]
    train_gpu = available_gpu_indices[1]
    model_device = "/device:GPU:" + str(model_gpu)
    train_device = "/device:GPU:" + str(train_gpu)
    print(f"model_device: {model_device}")
    print(f"train_device: {train_device}")

    # Test parameters
    model_type = "WaveNet_LSTM"
    exper_ids = pd.read_csv("./results/exper_ids.csv")  # 准备好的实验ID

    args = evaluating_args()
    data_name = args.data_name  # 要测试的数据名称
    stride = args.stride
    print(f"data_name: {data_name}, stride: {stride}")

    # Load pm25 data
    date = "20191125-2028"
    standard = "USA"
    window_size = 7200
    df_pm25_iaqi = pd.read_pickle(f"./data/{date}/pm25_iaqi_{standard}.pkl")
    df_pm25_level_polygonal = pd.read_pickle(
        f"./data/{date}/pm25_level_polygonal_{standard}.pkl")

    # set stride to 1, shuffle to False!
    window_sequence_test, _ = load_data(
        data_name, window_size=window_size, stride=1, batch_size=32, shuffle=False, standard="USA"
    )
    num_samples = len(window_sequence_test.index)
    num_batches = len(window_sequence_test)

    all_channels = True if data_name[-3:] == "all" else False
    if all_channels:
        multi_sequences, multi_ys = [], []
        for i in range(4):
            _data_name = "pm25" + f"_{i}"
            sequence = df_pm25_iaqi[_data_name].to_numpy()
            y = df_pm25_level_polygonal[_data_name].to_numpy()
            multi_sequences.append(sequence)
            multi_ys.append(y)
        sequence, y = np.asarray(multi_sequences), np.asarray(multi_ys)
    else:
        sequence = df_pm25_iaqi[data_name].to_numpy()
        y = df_pm25_level_polygonal[data_name].to_numpy()

    # Prepare model
    loss = tf.keras.losses.MeanSquaredError()  # "mse"
    metrics = [  # "mae", "mse", "mape", "msle"
        tf.keras.metrics.MeanAbsoluteError(),
        tf.keras.metrics.MeanSquaredError(),
        tf.keras.metrics.MeanAbsolutePercentageError(),
        tf.keras.metrics.MeanSquaredLogarithmicError()
    ]

    def create_model(data_name):
        if data_name == "pm25_all":
            model = WaveNet_LSTM_ver2021(input_shape=(
                window_size, 1), attention_type="MyAttention")
        else:
            model = WaveNet_LSTM(input_shape=(
                window_size, 1), attention_type="MyAttention")
        return model

    with tf.device(model_device):
        model = create_model(data_name)
        # model.summary()
        model.compile(Adam(lr=0.00001), loss=loss, metrics=metrics)

    # Different weights for different stride
    exper_id = exper_ids[(exper_ids["data_name"] == data_name) & (
        exper_ids["stride"] == stride)]["exper_time"].values[0]

    prefix = os.path.join("~", "Documents", "DeepLearningData", "AirMonitor")
    subfix = os.path.join(model_type, data_name, "_".join(
        ["stride", str(stride)]), exper_id)

    ckpt_dir = os.path.expanduser(os.path.join(prefix, "ckpts", subfix))
    ckpt_name = sorted(os.listdir(ckpt_dir))[-1]

    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    print(ckpt_path)

    if os.path.exists(ckpt_path):
        print("Model ckpt found! Loading: %s" % ckpt_path)
        model.load_weights(ckpt_path)
    else:
        raise OSError(f'Invalid ckpt_path: "{ckpt_path}!"')

    # Evaluation
    with tf.device(train_device):
        predictions = model.predict(window_sequence_test, verbose=1)
    save_path = f"./results/{data_name}-{stride}-predictions.npy"
    with open(save_path, 'wb') as f:
        np.save(f, predictions)
    print(f"predictions saved to: {save_path}")
    print(f"y shape: {y.shape}")
    print(f"num_samples: {num_samples}")
    print(f"predictions shape: {predictions.shape}")

    if data_name != "pm25_all":
        # Align y and predictions
        y_np = y[window_size-1:]
        pred = predictions[:-10]

        # Test results
        mae = 0
        for i in range(len(y_np)):
            mae += np.abs(y_np[i]-pred[i])
        mae /= len(y_np)
        print(f"Test mae: {mae}")

        mse = 0
        for i in range(len(y_np)):
            mse += np.square(y_np[i]-pred[i])
        mse /= len(y_np)
        print(f"Test mse: {mse}")

        with open("./results/test_results.csv", "a") as f:
            f.write(','.join([data_name, str(stride), str(
                mse[0]), str(mae[0]), exper_id]) + '\n')
    else:
        pass  # TODO


if __name__ == "__main__":
    main()
