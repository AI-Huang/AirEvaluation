#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jan-03-19
# @Update  : Aug-09-20 21:03
# @Author  : Kan Huang (kan.huang@connect.ust.hk)

import os
import sys
import re
import io
import time
from datetime import datetime
import csv
from optparse import OptionParser
import serial
from data_utils import makedirs_exist_ok
from sensor import Sensor

# 数据文件夹的命名规则 八位日期-四位时间
DIR_TIME_FORMAT = "%Y%m%d-%H%M"

# csv 表格时间数据的格式
TIME_FORMAT = "%Y-%m-%d %H:%M:%S %p"


sensor_ports = {
    0: "/dev/tty.wchusbserial14120",
    1: "/dev/tty.wchusbserial14130",
    2: "/dev/tty.wchusbserial141140",
    3: "/dev/tty.wchusbserial141110",
    4: "COM4"
}


def sensor_read_multithread_test():
    """TODO 多线程读取多个传感器数据
    多线程传入的是参数元组的列表
    最后用键盘ctrl c中断的，所有线程会一起中断吗？
    """
    import threadpool
    # thread_pool = threadpool.ThreadPool(4)
    # requests = threadpool.makeRequests(
    #     sensor_read, sensor_array)  # 不需要callback
    # [thread_pool.putRequest(req) for req in requests_]
    # thread_pool.wait()


def cmd_parser():
    parser = OptionParser()
    parser.add_option('--sensor_no', type='int', dest='sensor_no',
                      action='store', default=0, help='sensor_no, sensor number')
    parser.add_option('--verbose', type='int', dest='verbose',
                      action='store', default=1, help='verbose, if verbose>0, terminal will print PM data')

    args, _ = parser.parse_args(sys.argv[1:])
    return args


def SensorTest():
    """读取一个端口的传感器的数据
    """
    options = cmd_parser()
    sensor = Sensor(no=options.sensor_no, port=sensor_ports[options.sensor_no])
    sensor.connect()
    data_dir = os.path.join(
        "data", datetime.now().strftime(DIR_TIME_FORMAT))
    makedirs_exist_ok(data_dir)

    # handle a csv file
    headers = ['Time', 'PM25', 'PM10']
    dt = datetime.now()
    data_filepath = os.path.join(
        data_dir, "sensor" + str(sensor.no) + ".csv")
    if not os.path.isfile(data_filepath):
        with open(data_filepath, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, lineterminator=os.linesep)
            csv_writer.writerow(headers)
            csv_file.close()
    csv_file = open(data_filepath, 'a+', newline='')
    csv_writer = csv.writer(csv_file, lineterminator=os.linesep)

    # keeping reading port data and write them into csv file
    try:
        DEBUG = False
        while True:
            byte_counter = 0
            head_flag = 0
            msg = []
            # single msg
            while byte_counter < 10:
                # 读取一整组从msg head开始的字节
                binary_msg_byte = sensor.serial.readline(1)  # read 1 byte
                if DEBUG:
                    print(f"binary_msg_byte:{binary_msg_byte}")
                decimal_msg_byte = []
                for b in binary_msg_byte:  # iterate it into int
                    decimal_msg_byte.append(b)
                decimal_msg_byte = decimal_msg_byte[0]
                if DEBUG:
                    print(f"decimal_msg_byte:{decimal_msg_byte}")
                if decimal_msg_byte == 170:
                    # print("Got msg head!")
                    head_flag = 1
                if head_flag:
                    msg.append(decimal_msg_byte)
                    byte_counter += 1
            if DEBUG:
                print(f"msg:{msg}")

            ret, pm25, pm10 = sensor.extract_message(msg)
            if ret == -1:
                pass
            elif ret == 1:
                dt = datetime.now()
                t = dt.strftime(TIME_FORMAT)
                if options.verbose > 0:
                    print(t)
                    # print("PM25:%f, PM10:%f" % (pm25, pm10))  # TODO
                    # TODO
                    print(f"Sensor_no: {sensor.no} PM25: {pm25}, PM10: {pm10}")
                csv_writer.writerow([t, pm25, pm10])

    except KeyboardInterrupt:
        print('Program ending...')
        csv_file.close()
        print('File handle closed.')
        sensor.disconnect()
        print('Serial port closed!')


def main():
    SensorTest()


if __name__ == '__main__':
    main()
