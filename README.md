## AirEvaluation

AirEvaluation，GreenEyes 空气监控和阈值判断及其预测系统。

### Training

Demo run:

```bash
python train.py --data_type=pm25 --model_type="WaveNet_LSTM" --attention_type="official" --sensor_index=0 --shuffle=True --stride=10
```

where `sensor_index=0` means the No. 0 sensor's data is chosen;

`stride=10` means the stride for the sliding window is 10.

### Evaluation

Run:

```bash
python evaluate.py --stride=10
```
