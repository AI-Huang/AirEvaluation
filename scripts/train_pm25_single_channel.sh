#!/bin/bash
# 2022-2-12 18:05:16 Hong Kong

hparam_sensor_index="0 1 2 3"
hparam_stride="10 5 2"
attention_type="official" # "MyAttention"
for sensor_index in $hparam_sensor_index; do
    for stride in $hparam_stride; do
        echo "sensor_index: $sensor_index; stride: $stride; attention_type: $attention_type."
        python train.py --data_type=pm25 --sensor_index=$sensor_index --shuffle=True --stride=$stride --attention_type=$attention_type --output_dir="."
    done
done
