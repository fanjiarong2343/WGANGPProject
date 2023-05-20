import os
import sys
import numpy as np
import tensorflow as tf
from sequence.utils import *
from sequence.data_process import *
from nn.wd_trainer import WassersteinTrainer
from nn.scale_model import ScaleModel
from config import *
from nn.optimizer import Optimizer


if __name__ == '__main__':
    format_dir = 'data/readelf/format_set'
    output_dir = 'data/readelf'

    c_dir = output_dir + '/c_source'
    cmd = "ls {}".format(c_dir)
    result, _ = subprocess_call(cmd)
    split = result[0].split()
    # if there exists no files
    if len(split) == 0:
        print("Note: There exists no files!")
    print("文件数量：", len(split))
    
    i = 1
    while i <= 8751:
        file = split[i - 1]
        file_name = file.rstrip()
        file_path = "{}/{}".format(c_dir, file_name)
        outfile_name = 'org_elf:id:{}'.format(str(i).zfill(4))
        outfile_path = os.path.join(format_dir, outfile_name)
        print('file_path:', file_path)
        gcc_cmd = "gcc -Wall -s {} -o {}".format(file_path, outfile_path)
        # gcc -Wall -s c_source/id:0799.c -o format_set/org_elf:id:0001
        gcc_result, _ = subprocess_call(gcc_cmd)
        # if gcc_result is not None:
        i += 1

