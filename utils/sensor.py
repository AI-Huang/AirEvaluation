#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jan-03-19
# @Update  : Aug-09-20 21:03
# @Author  : Kan Huang (kan.huang@connect.ust.hk)
# @RefLink : https://pyserial.readthedocs.io/en/latest/pyserial_api.html

"""Sensor util class.
Get PM2.5/10 data from the sensor in real time via BLE-USB port.

# Requirements
    pyserial>=3.4
# Usage

"""
import os
import sys
import re
import io
import time
from datetime import datetime
import csv
import serial
from data_utils import makedirs_exist_ok

# 数据文件夹的命名规则 八位日期-四位时间
DIR_TIME_FORMAT = "%Y%m%d-%H%M"

# csv 表格时间数据的格式
TIME_FORMAT = "%Y-%m-%d %H:%M:%S %p"


def compute_checksum(int8tocheck):
    """calculate checksum
    """
    checksum = 0
    for num in int8tocheck:
        checksum += num
    checksum = checksum & 0xFF
    return checksum


def checksum_test():
    byte_array = [0x40, 0x00, 0x57, 0x00, 0x9E, 0x8B]
    checksum = compute_checksum(byte_array)
    print(checksum)
    print("%x" % checksum)


class Sensor(object):
    """USB sensor object
    """

    def __init__(self, no, port):
        self.no = no  # sensor number
        self.port = port
        self.serial = None  # Serial object

    def connect(self):
        port = self.port
        self.serial = serial.Serial(port=port,
                                    baudrate=9600,
                                    bytesize=serial.EIGHTBITS,
                                    parity=serial.PARITY_NONE,
                                    stopbits=serial.STOPBITS_ONE,
                                    timeout=5)  # 5s is connection timeout

    def disconnect(self):
        self.serial.close()
        self.serial = None

    def read(self):
        """TODO
        """
        pass

    def send(self, byte_list):
        """TODO not necessary to implement, alternate with serial assistant
        """
        byte_list = [0xAA, 0xB4, 0xAA, 0xAA, 0xAA]
        pass

    def extract_message(self, msg):
        """extract PM2.5 and PM10 message from the sensor
        """
        ret = 0
        if len(msg) is not 10:
            ret = -1
            return ret
        if msg[-1] is not 171:  # 0xAB
            ret = -1
            return ret
        byteHead = msg[0]
        byteCmd = msg[1]
        bytePM25L = msg[2]
        bytePM25H = msg[3]
        bytePM10L = msg[4]
        bytePM10H = msg[5]
        byteSen1ID = msg[6]
        byteSen2ID = msg[7]
        byteChecksum = msg[8]  # 52 correct
        byteTail = msg[9]

        byteToCheck = [bytePM25L, bytePM25H, bytePM10L,
                       bytePM10H, byteSen1ID, byteSen2ID]
        checkSum = compute_checksum(byteToCheck)

        ret = 1
        if byteChecksum is not checkSum:
            ret = -1
            print("Warning! Checksum incorrect! Suggest you to abort the data.")

        pm25 = (bytePM25H*256 + bytePM25L)/10.0
        pm10 = (bytePM10H*256 + bytePM10L)/10.0
        return ret, pm25, pm10


def SensorTest():
    """读取一个端口的传感器的数据
    """
    verbose = 1
    sensor = Sensor(no=0, port="COM3")  # for Windows
    sensor.connect()
    data_dir = os.path.join(
        "data", datetime.now().strftime(DIR_TIME_FORMAT))
    makedirs_exist_ok(data_dir)

    # handle a csv file
    headers = ['Time', 'PM25', 'PM10']
    dt = datetime.now()
    data_filepath = os.path.join(data_dir, "Sensor" + str(sensor.no) + ".csv")
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
                if verbose > 0:
                    print(t)
                    print("PM25:%f, PM10:%f" % (pm25, pm10))
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
