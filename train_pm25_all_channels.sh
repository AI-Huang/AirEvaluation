#!/bin/bash
# 10, 5, 2
python train.py --data_name=pm25_all --shuffle=True --stride=5 --log_prefix="." 