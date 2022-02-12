# AirEvaluation

GreenEyes: An Air Quality Level Fitting Model based on WaveNet

AirMonitor data

## Dataset Description

数据维度：

220696 x 1

### 数据格式

#### 数据文件夹的命名规则

八位日期-四位时间

```Python
DIR_TIME_FORMAT = "%Y%m%d-%H%M"
```

#### 时间戳格式

csv 表格时间数据的格式

```Python
dt = datetime.now()
date = dt.strftime("%Y%m%d-%H%M%S-%p")
t = dt.strftime("%Y-%m-%d %H:%M:%S %p")
```

例如：2019-11-25 20:28:24 PM

#### 时间格式转换

```Python
datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S %p")
```
