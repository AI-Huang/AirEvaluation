#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jul-16-20 13:29
# @Update  : Oct-20-20 21:06
# @Author  : Kan Huang (kan.huang@connect.ust.hk)

"""Data utils test
"""
import os
import time
import numpy as np
import pandas as pd
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets.data_utils import iaqi, rel_time


def nan_test():
    """nan_test
    """
    # Load pm25 data
    date = "20191125-2028"
    df_pm25 = pd.read_pickle(
        f"./data/{date}/pm25.pkl")  # pandas 格式的数据
    standard = "USA"
    df_pm25_iaqi = pd.read_pickle(
        f"./data/{date}/pm25_iaqi_{standard}.pkl")
    df_pm25_level_polygonal = pd.read_pickle(
        f"./data/{date}/pm25_level_polygonal_{standard}.pkl")

    time_indices = np.asarray([int(time.mktime(_))
                               for _ in df_pm25["time"]])  # 从1970年开始的秒数
    time_indices = rel_time(time_indices)

    data_list = ["pm25_0", "pm25_1", "pm25_2", "pm25_3"]

    # check if there is an nan number
    for _ in data_list:
        data = df_pm25[_].to_numpy()
        a = np.sum(data)
        if np.isnan(a):
            raise "Error!"

    for _ in data_list:
        data = df_pm25_iaqi[_].to_numpy()
        a = np.sum(data)
        if np.isnan(a):
            raise "Error!"

    # test df_pm25_level
    for _ in data_list:
        data = df_pm25_level_polygonal[_].to_numpy()
        a = np.sum(data)
        if np.isnan(a):
            raise "Error!"


def main():
    iaqi("pm25", 0)
    iaqi("pm25", 0, USA=True)
    iaqi("pm25", -1)
    iaqi("pm25", 12)
    iaqi("pm25", 34.9)
    iaqi("pm25", 35)
    iaqi("pm25", 74.99)
    iaqi("pm25", 114)
    iaqi("pm25", 75)
    assert 35 < 75.0 <= 75
    assert not 35 < 76 <= 75

    # nan_test()


def plot_iaqi_function():
    plt.figure(figsize=(5.0, 5.0), dpi=100)
    a = np.linspace(0, 1000, 1000)
    [iaqi("pm25", _) for _ in a]
    iaqi_pm25 = np.array([_[0] for _ in [iaqi("pm25", _) for _ in a]])
    iaqi_pm10 = np.array([_[0] for _ in [iaqi("pm10", _) for _ in a]])
    plt.xlim(0, 800)
    plt.ylim(0, 800)
    plt.xlabel("Category data value ($\mu g/m^3$)")
    plt.ylabel("IAQI")
    plt.plot(a, label="IAQI")
    plt.plot(iaqi_pm25, label="$PM_{2.5}$ IAQI")
    plt.plot(iaqi_pm10, label="$PM_{10}$ IAQI")
    plt.legend()
    plt.tight_layout()
    plt.grid()


if __name__ == "__main__":
    main()
