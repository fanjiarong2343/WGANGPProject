import os
import sys
import numpy as np
import tensorflow as tf
from sequence.data_process import *
from sequence.utils import *
from nn.wd_trainer import WassersteinTrainer
from nn.scale_model import ScaleModel
from config import *
from nn.optimizer import Optimizer


def build_edgeset():
    format_set = FORMAT_DIR
    edge_set = EDGE_DIR

    # 1. 新建EDGE_DIR，将FORMAT_DIR的全部测试用例复制至EDGE_DIR
    if not os.path.exists(edge_set):
        os.mkdir(edge_set)
    cp_cmd = "cp -r {}/. {}".format(format_set, edge_set)
    _ = subprocess_call(cp_cmd)

    # 2. 找到4个generate文件
    output_dir = OUTPUT_DIR
    directories = os.listdir(output_dir)
    filter_dirs = []
    for bits in ['2bits', '4bits', '8bits', '16bits']:
        generate_list = list(filter(lambda x: x.find('generate') >= 0 and x.find(
            bits) >= 0, directories))
        generate_list.sort(reverse=True)  # list.sort(key=None, reverse=False) 当reverse=True时为降序排列，reverse=False为升序排列
        filter_dirs.append(generate_list[0])

    #  2 -> 4 -> 8 -> 16
    # 1000：100：100：100
    count = [1000, 100, 100, 100]
    for i in range(len(filter_dirs)):
        path = os.path.join(output_dir, filter_dirs[i])
        # print("generate_path: ", path)
        ls_cmd = "ls {} | head -n {}".format(path, count[i])
        r, _ = subprocess_call(ls_cmd)
        split = r[0].split()
        # print('split:', split)
        for s in split:
            cp_cmd = "cp {}/{} {}/{}".format(path, s, EDGE_DIR, s+str(i))
            _, _ = subprocess_call(cp_cmd)


if __name__ == '__main__':
    build_edgeset()
