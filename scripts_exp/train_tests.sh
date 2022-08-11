# #!/bin/bash
# # 2022-2-12 18:05:16 Hong Kong

# Normal training
# python train.py --model_name "WaveNet_LSTM" --attention_type=official --data_type=pm25 --sensor_index=0 --shuffle=True --stride=10 --output_dir="./output"

# No attention
python train.py --model_name "WaveNet_LSTM" --no-attention --data_type=pm25 --sensor_index=0 --shuffle=True --stride=10 --output_dir="./output"
