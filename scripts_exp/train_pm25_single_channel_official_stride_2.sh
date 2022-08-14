#!/bin/bash
# 2022-2-16 23:57:59 Hong Kong

hparam_sensor_index="0 1" # "0 1 2 3"
hparam_stride="10 5" # "10 5 2"
attention_type="official" # "MyAttention"
hparam_window_size="7200 3600"
for sensor_index in $hparam_sensor_index; do
    for stride in $hparam_stride; do
        for window_size in $hparam_window_size; do
        echo "sensor_index: $sensor_index; stride: $stride; attention_type: $attention_type."
            python train.py --data_type=pm25 --sensor_index=$sensor_index --window_size=$window_size --shuffle=True --stride=$stride --attention_type=$attention_type --output_dir="./output"
        done
    done
done
