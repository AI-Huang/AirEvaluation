#!/bin/bash
# 2022-2-12 18:20:28

output_dir="~/Documents/Exper-no-attention"

hparam_sensor_index="0 1 2 3"
hparam_stride="10 5" # 2

for stride in $hparam_stride; do

    for sensor_index in $hparam_sensor_index; do
        echo "sensor_index: $sensor_index; stride: $stride; --no-lstm"
        python train.py --model_name "WaveNet_LSTM" --no-lstm --data_type=pm25 --sensor_index=$sensor_index --shuffle=True --stride=$stride --output_dir="./output"

        # TODO send email from xx to yy when a run finished
        # python send_exper_email.py --stride --sensor_index --cur_time
    done

done
