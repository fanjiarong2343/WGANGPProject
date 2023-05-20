import os
import sys
import numpy as np
import tensorflow as tf
from sequence.data_process import *
from nn.wd_trainer import WassersteinTrainer
from nn.scale_model import ScaleModel
from config import *
from nn.optimizer import Optimizer
import requests
from bs4 import BeautifulSoup
import re


headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36'}

# 已访问过的URL集合
url_set = {}


def is_visit(url_set, url):
    if url_set.get(url) is None:
        return False
    else:
        return True


def c_source(url, i):
    c = requests.get(url, headers=headers, timeout=30)
    c.encoding = "utf-8"
    html = c.text

    soup = BeautifulSoup(html, features="html.parser")

    items = soup.find_all("code", attrs={"class": "language-c"})
    
    for item in items:
        c_source = item.text.strip()
        with open("data/readelf/c_source/id:{}.c".format(str(i).zfill(4)), "w", encoding='utf-8') as f:
            f.write(c_source)
        f.close()
        i += 1
    return i


def neighbor(url):
    c = requests.get(url, headers=headers, timeout=30)
    c.encoding = "utf-8"
    html = c.text

    neighbor_url = []
    soup = BeautifulSoup(html, features="html.parser")
    a_item = soup.find_all("a")
    for a in a_item:
        href = a.get("href")
        if href is None:
            continue
        if href.find("blog.csdn.net") != -1 and href.find("article") != -1 and href.find("details") != -1:
            neighbor_url.append(href)            
    return neighbor_url


# 输入一个起始点 URL
def BFS(url, i):
    # 创建队列
    queue = []
    
    # queue.append(url)
    url_set[url] = 1
    neighbor_url = neighbor(url)
    # 遍历neighbor_url中的元素
    for n_url in neighbor_url:
        # 如果n_url没访问过
        if not is_visit(url_set, n_url):
            queue.append(n_url)

    # 当队列不空的时候
    while len(queue) > 0:
        # 将队列的第一个元素读出来
        url = queue.pop(0)
        print("url:", url)
        new_i = c_source(url, i)
        neighbor_url = []
        if new_i > i:
            neighbor_url = neighbor(url)
            i = new_i
        
        # 加入url_set表示url我们访问过
        url_set[url] = 1

        # 遍历neighbor_url中的元素
        for n_url in neighbor_url:
            # 如果n_url没访问过
            if not is_visit(url_set, n_url):
                queue.append(n_url)
        if i > 15000:
            break


if __name__ == '__main__':
    # 广度优先搜索
    # i = 1702
    i = 8006

    url = "https://blog.csdn.net/nav/job"
    BFS(url, i)
