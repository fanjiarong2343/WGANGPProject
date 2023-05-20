import os
import sys
import numpy as np
import tensorflow as tf
from sequence.data_process import *
from nn.wd_trainer import WassersteinTrainer
from nn.scale_model import ScaleModel
from config import *
from nn.optimizer import Optimizer
from argparse import ArgumentParser


def revise_line(lines, parameter, type, value):
    l = len(lines)
    for i in range(l):
        if lines[i].startswith(parameter):
            if type == "str":
                lines[i] = parameter + " = '" + str(value) + "'\n"
            else:
                lines[i] = parameter + ' = ' + str(value) + '\n'
            break
    return lines


def revise(argvs):
    # 修改 config.py 文件
    with open('config.py', 'r', encoding='utf-8') as old_config:
        # readlines返回每行字符串组成的list
        lines = old_config.readlines()
        for i in range(1, len(argvs), 3):
            parameter = argvs[i]
            type = argvs[i+1]
            value = argvs[i+2]
            lines = revise_line(lines, parameter, type, value)

    with open('config.py', 'w', encoding='utf-8') as new_config:
        new_config.writelines(lines)


if __name__ == '__main__':
    '''
    执行命令格式:
        python revise_config.py 待修改的参数名 参数类型 参数值 ...
        eg. python revise_config.py PRECISION str 8
    '''
    argvs = sys.argv
    # argvs = ['', 'PRECISION', 'str', '8']
    revise(argvs)