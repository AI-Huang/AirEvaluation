#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jan-12-19
# @Author  : Kan Huang (kan.huang@connect.ust.hk)


import os
import time
import requests
from bs4 import BeautifulSoup


def download_page(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.80 Safari/537.36'
    }
    data = requests.get(url, headers=headers).content
    return data


def main():
    query_url = "http://aqicn.org/city/some_a_city"
    try:
        # data file
        dataFile = open("aqicn-20190112-23.csv", 'a+')
        counter = 0
        while True:
            query_html = download_page(query_url).decode('utf-8')
            query_soup = BeautifulSoup(query_html, "lxml")
            updateSpan = query_soup.find("span", {"id": "aqiwgtutime"})
            updateText = updateSpan.text
            print(updateText)
            cur_pm25 = query_soup.find("td", {"id": "cur_pm25"})
            cur_pm25_text = cur_pm25.text
            print(cur_pm25_text)
            cur_pm10 = query_soup.find("td", {"id": "cur_pm10"})
            cur_pm10_text = cur_pm10.text
            print(cur_pm10_text)
            time.sleep(1)
            # dataFile.write(updateText + ',' + cur_pm25_text + ',' + cur_pm10_text + '\n')
            if counter is 0:
                dataFile.write(updateText + ',' + cur_pm25_text +
                               ',' + cur_pm10_text + '\n')
            counter += 1
            if counter is 60:
                counter = 0
            # time.sleep(60*15) # seconds
    except KeyboardInterrupt:
        dataFile.close()
        print('Program ending...')


if __name__ == '__main__':
    main()
