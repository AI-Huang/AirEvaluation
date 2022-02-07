# --model_type=WaveNet_LSTM
# python train.py --shuffle=True --activation=relu --batch_norm=True --attention_type=official
# python train.py --shuffle=True --activation=relu --batch_norm=False --attention_type=official
# python train.py --shuffle=True --stride=5
# python train.py --shuffle=True --stride=2

params_sensor_index="1 2 3 4"
params_stride="10 5 2"
for sensor_index in $params_sensor_index;
do
for stride in $params_stride;
do
echo "sensor_index: $sensor_index, stride: $stride."
python train.py --data_type=pm25 --sensor_index=$sensor_index --shuffle=True --stride=$stride
done
done