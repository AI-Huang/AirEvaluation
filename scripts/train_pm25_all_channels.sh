#!/bin/bash
hparam_stride="10 5 2" # 10, 5, 2
for sensor_index in $hparam_sensor_index; do
    python train.py --data_name=pm25_all --shuffle=True --stride=5 --log_prefix="."
done