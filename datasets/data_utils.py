#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Nov-04-19 19:50
# @Update  : Oct-18-20 21:28
# @Author  : Kan Huang (kan.huang@connect.ust.hk)

import os
import csv
import time
import numpy as np
import pandas as pd


# csv表格时间数据的格式
TIME_FORMAT = "%Y-%m-%d %H:%M:%S %p"

# Standards
CN_IAQI_STANDARD = {
    "pm25": [0, 35, 75, 115, 150, 250, 350, 500],
    "pm10": [0, 50, 150, 250, 350, 420, 500, 600]
}


USA_IAQI_STANDARD = {
    "pm25": [0, 12.1, 35.5, 55.5, 150.5, 250.5, 500.4],
    "pm10": [0, 55, 155, 255, 355, 425, 604]
}


def rel_time(time_indices, to_hour=True):
    """relative time in seconds for a time series
    Input:
        time_indices: in numpy array
    """
    t0 = time_indices[0]
    time_indices -= t0
    # relative time in hour(float)
    if to_hour:
        time_indices = np.divide(time_indices, 3600)
    return time_indices


def iaqi(value, category: str, standard="USA"):
    """按照参考文献对IAQI的定义计算IAQI，计算对象为单个数据
    Inputs:
        category: PM2.5 or PM10
        value: data value
        standard: which IAQI standard to use, "China" or "USA", "China" by default
    Return:
        iaqi: iaqi score, float
        iaqi_level: iaqi level, int
    """
    if standard not in ["China", "USA"]:
        raise ValueError("standard must be one of: " + '["China", "USA"]')

    # CN and USA standards share the same IAQI thresholds
    iaqi_thresholds = [0, 50, 100, 150, 200, 300, 500]  # USA
    iaqi_thresholds = [0, 50, 100, 150, 200, 300, 400, 500]  # China

    if standard == "China":
        d = CN_IAQI_STANDARD
    elif standard == "USA":
        d = USA_IAQI_STANDARD

    if category not in d.keys():
        raise ValueError("Pollution type must be one of:\n" + "pm25, pm10")

    bp_thresholds = d[category]  # 污染物浓度限值
    # 均按照左开右闭区间处理
    if value <= bp_thresholds[0]:
        return iaqi_thresholds[0]
    if value > bp_thresholds[-1]:
        return iaqi_thresholds[-1]
    for i, t in enumerate(bp_thresholds):
        if bp_thresholds[i] < value <= bp_thresholds[i+1]:  # 临界值往下算
            iaqi = (value-bp_thresholds[i]) /\
                (bp_thresholds[i+1]-bp_thresholds[i]) *\
                (iaqi_thresholds[i+1]-iaqi_thresholds[i]) +\
                iaqi_thresholds[i]
            return iaqi


def iaqi_level(value, category: str, standard="USA"):
    """按照参考文献对IAQI的定义计算IAQI，计算对象为单个数据
    Inputs:
        value: data value.
        category: PM2.5 or PM10.
        standard: which IAQI standard to use, "China" or "USA", default "USA".
    Return:
        iaqi: iaqi score, float
        iaqi_level: iaqi level, int
    """
    if standard not in ["China", "USA"]:
        raise ValueError("standard must be one of: " + '["China", "USA"]')

    # CN and USA standards share the same IAQI thresholds
    iaqi_thresholds = [0, 50, 100, 150, 200, 300, 400, 500]

    if standard == "China":
        d = CN_IAQI_STANDARD
    elif standard == "USA":
        d = USA_IAQI_STANDARD

    if category not in d.keys():
        raise ValueError("Pollution type must be one of:\n" + "pm25, pm10")

    bp_thresholds = d[category]  # 污染物浓度限值
    # 均按照左开右闭区间处理
    if value <= bp_thresholds[0]:
        return 1
    if value > bp_thresholds[-1]:
        return 6
    for i, t in enumerate(bp_thresholds):
        if bp_thresholds[i] < value <= bp_thresholds[i+1]:  # 临界值往下算
            iaqi = (value-bp_thresholds[i]) /\
                (bp_thresholds[i+1]-bp_thresholds[i]) *\
                (iaqi_thresholds[i+1]-iaqi_thresholds[i]) +\
                iaqi_thresholds[i]
            iaqi_level = i + 1
            if iaqi > iaqi_thresholds[5]:
                iaqi_level = 6
            return iaqi_level


def iaqi_np(data, category: str, standard="USA"):
    """numpy 向量化方法计算一个sequence数据的IAQI
    # Arguments:
        data: numpy array, sequence of data.
        category: PM2.5 or PM10.
        standard: which IAQI standard to use, "China" or "USA", default "USA".
    # Return:
        iaqi: iaqi score, float
        iaqi_level: iaqi level, int
    """
    if standard not in ["China", "USA"]:
        raise ValueError("standard must be one of: " + '["China", "USA"]')

    # CN and USA standards share the same IAQI thresholds
    iaqi_thresholds = [0, 50, 100, 150, 200, 300, 400, 500]

    if standard == "China":
        d = CN_IAQI_STANDARD
    elif standard == "USA":
        d = USA_IAQI_STANDARD

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    iaqi = np.zeros(data.shape) - 1  # -1s

    bp_thresholds = d[category]  # 污染物浓度限值
    # 均按照左开右闭区间处理
    iaqi = np.where(data <= bp_thresholds[0], 1, -1)
    iaqi += np.where(data > bp_thresholds[-1], 6, -1)
    for i, t in enumerate(bp_thresholds):
        ttt = (data-bp_thresholds[i]) /\
            (bp_thresholds[i+1]-bp_thresholds[i]) *\
            (iaqi_thresholds[i+1]-iaqi_thresholds[i]) +\
            iaqi_thresholds[i]
        iaqi += np.where(bp_thresholds[i] < data <=
                         bp_thresholds[i+1], ttt, -1)  # 临界值往下算

    assert -1 not in iaqi
    return iaqi


def load_csv(csv_path):
    """load PM data from a csv file
    """
    with open(csv_path, "r") as f:
        f_reader = csv.reader(f, delimiter=',')
        data = []
        for row in f_reader:
            data.append(row)
        data.pop(0)  # remove header
        times, pm25s, pm10s = [], [], []
        for _, row in enumerate(data):
            t = time.strptime(row[0], TIME_FORMAT)  # 转换为 time.struct_time 对象
            pm25 = float(row[1])
            pm10 = float(row[2])
            times.append(t)
            pm25s.append(pm25)
            pm10s.append(pm10)

        return times, pm25s, pm10s


def load_csv_test(data_dir="./data/20191125-2028", csv_file="sensor0.csv"):

    csv_file = os.path.join(data_dir, csv_file)
    times, pm25s, pm10s = load_csv(csv_file)
    import matplotlib.pyplot as plt
    plt.plot(pm25s[0])
    plt.show()

    return times, pm25s, pm10s


def data_intersect(time_array, data_array):
    """intersect several sequence data on the time axis
    """
    num_axis = len(time_array)
    intersect_buffer, count = {}, {}
    # init with first time, data sequence
    for i, t in enumerate(time_array[0]):
        intersect_buffer[t] = [data_array[0][i]]
        count[t] = 1

    # find data which locates at the same time in the following files
    for i in range(num_axis):
        if i == 0:
            continue
        time, data = time_array[i], data_array[i]
        for i, t in enumerate(time):
            if t in intersect_buffer.keys():  # O(n(n-1)/2)
                intersect_buffer[t].append(data[i])
                count[t] += 1

    # 过滤缺数据的时间点
    intersect_time = []
    intersect_data = [[], [], [], []]
    for t in time_array[0]:
        if count[t] == num_axis:
            intersect_time.append(t)
            for i in range(num_axis):
                intersect_data[i].append(intersect_buffer[t][i])

    print(f"len_intersect:{len(intersect_time)}")

    return intersect_time, intersect_data


def save_intersected_data(data_dir="./data/20191125-2028"):
    """保存统一时间轴后的数据，包括PM2.5数据和PM10数据
        Input:
            data_dir: e.g., "./data/20191125-2028"
    """
    csv_files = ["sensor0.csv",
                 "sensor1.csv",
                 "sensor2.csv",
                 "sensor3.csv"]
    num_sensors = len(csv_files)

    time_array, pm25_array, pm10_array = [], [], []
    for f in csv_files:
        time, pm25, pm10 = load_csv(os.path.join(data_dir, f))
        time_array.append(time)
        pm25_array.append(pm25)
        pm10_array.append(pm10)

    time, data = data_intersect(time_array, pm25_array)

    print("123")
    print(len(data[0]))
    print(len(data[1]))
    print(len(data[2]))
    print(len(data[3]))

    df_pm25 = pd.DataFrame({
        'time': time,
        'pm25_0': data[0],
        'pm25_1': data[1],
        'pm25_2': data[2],
        'pm25_3': data[3],
    })
    df_pm25.to_pickle(os.path.join(data_dir, "pm25.pkl"))

    time, data = data_intersect(time_array, pm10_array)

    print("123")
    print(len(data[0]))
    print(len(data[1]))
    print(len(data[2]))
    print(len(data[3]))

    df_pm10 = pd.DataFrame({
        'time': time,
        'pm10_0': data[0],
        'pm10_1': data[1],
        'pm10_2': data[2],
        'pm10_3': data[3],
    })
    df_pm10.to_pickle(os.path.join(data_dir, "pm10.pkl"))


def label_edges(manual_points, time_indices, data):
    """label_edges, 根据手动 label 的 manual_points 精准标记边缘点 edges
    Inputs:
        manual_points: list like variable containing edge points, time in Hour;
        time_indices: time axis data, in Hour;
        data: for example, df_pm25_level.
    Return:
    """
    edge_indices, edge_t = [], []

    #
    for i in range(0, len(manual_points), 2):
        _edge, count = 0, 0
        for j, t in enumerate(time_indices):
            if manual_points[i] <= t < manual_points[i+1]:
                if data[j] == 2:
                    _edge_indices, _edge_t = j, t
                    if count == 0:
                        edge_indices.append(_edge_indices)
                        edge_t.append(_edge_t)
                        count += 1
            elif count == 1:
                edge_indices.append(_edge_indices)
                edge_t.append(_edge_t)
                break
    return edge_indices, edge_t


def flatten_edges(data, time_indices, edges):
    """flatten_edges, flatten the data line into rectangles, the returned data will be used for generating polygonal lines
    Inputs:
        data: numpy array data, for example, pm25_iaqi_level;
        time_indices: time axis data;
        edge_points: list like variable containing edge points;
    """
    rectangle_lines = data.copy()
    for i in range(0, len(edges), 2):
        for j, t in enumerate(time_indices):
            if edges[i] <= j < edges[i+1]:
                rectangle_lines[j] = 2
            if i + 2 < len(edges):
                if edges[i+1] <= j < edges[i+2]:
                    rectangle_lines[j] = 1
    for j, t in enumerate(time_indices):
        if edges[-1] <= j:
            rectangle_lines[j] = 1
    return rectangle_lines


def create_polygonal(rectangle_lines, edge_indices, edge_time, time_in_hour):
    """create_polygonal, use rectangle lines to generate polygonal lines
    Inputs:
        rectangle_lines:
        edge_indices: list like variable containing edge points.
        edge_time: .
        time_in_hour: time axis data.
    """
    num_samples = len(time_in_hour)
    # polygonal_lines = rectangle_lines.copy() # leads to unmutable bug
    polygonal_lines = np.empty(num_samples)

    # add left and right paddings for edge_indices
    edge_indices = [0] + edge_indices.copy() + [num_samples-1]
    edge_time = [0] + edge_time.copy() + [time_in_hour[-1]]

    for i, _ in enumerate(edge_indices):
        if i == 0:
            continue

        slope = (rectangle_lines[edge_indices[i]]-rectangle_lines[edge_indices[i-1]]) /\
            (edge_time[i]-edge_time[i-1])

        initial_time = edge_time[i-1]
        initial = rectangle_lines[edge_indices[i-1]]

        # +1 to form a close interval
        for j in range(edge_indices[i-1], edge_indices[i]+1):
            delta_time = (time_in_hour[j] - initial_time)
            increase = slope * delta_time
            polygonal_lines[j] = increase + initial

    return polygonal_lines


def main():
    start = time.process_time()

    save_intersected_data(data_dir="./data/20191125-2028")  # 219989

    elapsed = (time.process_time() - start)
    print(f"Time used: {elapsed}s")  # Time used: 27.140625s

    # save_intersected_data(data_dir="./data/20200809-2147")
    # save_intersected_data(data_dir="./data/20200811-1532")  # N=1362641


if __name__ == "__main__":
    main()
