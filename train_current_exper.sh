#!/bin/bash
# 2022-2-12 18:20:28

python train.py --data_type=pm25 --model_type="WaveNet_LSTM" --attention_type="MyAttention" --sensor_index=0 --shuffle=True --stride=10

python train.py --data_type=pm25 --model_type="WaveNet_LSTM" --attention_type="official" --sensor_index=0 --shuffle=True --stride=10

