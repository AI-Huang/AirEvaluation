# 模型批量测试 shell 脚本
# 测试两个参数 dataname, stride
# 一共 5*3=15 组
python evaluate.py --data_name=pm25_0 --stride=10
python evaluate.py --data_name=pm25_0 --stride=5
python evaluate.py --data_name=pm25_0 --stride=2

# python evaluate.py --data_name=pm25_0 --stride=10