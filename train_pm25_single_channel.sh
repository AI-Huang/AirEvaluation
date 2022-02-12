# 2022-2-12 18:05:16


hparam_sensor_index="0 1 2 3"
hparam_stride="10 5 2"
for sensor_index in $param_sensor_index; do
    for stride in $param_stride; do
        echo "sensor_index: $sensor_index, stride: $stride."
        python train.py --data_type=pm25 --sensor_index=$sensor_index --shuffle=True --stride=$stride
    done
done
