python train.py --data_type=pm25 --model_type=="WaveNet_LSTM" --attention_type="MyAttention" --sensor_index=0 --shuffle=True --stride=10

param_sensor_index="0 1 2 3"
for sensor_index in $param_sensor_index; do
    python3 train.py --data_type=pm25 --sensor_index=$sensor_index --shuffle=True --stride=10 --epochs=100;
done