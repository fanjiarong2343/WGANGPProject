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


output_dir = OUTPUT_DIR
program_path = PROGRAM_PATH
parameter = PARAMETER
max_count = MAX_COUNT
# output_dir = 'data/jhead'
# program_path = output_dir + '/jhead'
# parameter = '-v @@'
# max_count = 8


def generate_dir():
    directories = os.listdir(output_dir)
    bits = '2bits'
    generate_list = list(filter(lambda x: x.find('generate') >= 0 and x.find(
        bits) >= 0, directories))
    generate_list.sort(reverse=True)  # list.sort(key=None, reverse=False) 当reverse=True时为降序排列，reverse=False为升序排列
    generate_dir = generate_list[0]
    print('generate_dir: ', generate_dir)
    return '/'+generate_dir+'/'


generate = generate_dir()

file_path = os.path.join(output_dir, 'dictkey_edge.npy')
dictkey_edge = np.load(file_path)


def aflshowmap(showmap_file):
    showmap_out = "showmap"
    cmd = "export AFL_MAP_SIZE=102400K; {} -o {} -- {} {}".format(SHOWMAP, showmap_out, program_path, parameter)
    cmd = cmd.replace("@@", showmap_file)
    print("afl-showmap:", cmd)
    _, _ = subprocess_call(cmd)
    return showmap_out


def bitmap(label, file_name):
    tmp_list = []

    content = open(file_name, "r", encoding='ISO-8859-1')  # 打开文件
    for line in content.readlines():
        line = line.strip()  # 去掉每行头尾空白
        edge = line.split(':')[0]
        count = int(line.split(':')[1])
        tmp_list.append((edge, count))
    content.close()  # 关闭文件

    bitmap = np.zeros(len(label))
    for j in tmp_list:
        if j[0] in label:
            bitmap[label.index(j[0])] = j[1]

    bitmap = bitmap / max_count  # 归一化
    return bitmap


def compare_single(i, out_path):
    np.set_printoptions(threshold=np.inf)

    case = "original"  # original random concat
    condition_file = "c_{}.npy".format(case)
    condition_path = os.path.join(out_path, condition_file)
    condition = np.load(condition_path)
    # print('condition[]:', condition[i])

    file_name = out_path + "all/{}/gen:id:{}".format(case, str(i).zfill(4))
    showmap_out = aflshowmap(file_name)
    bitmap_c = bitmap(dictkey_edge.tolist(), showmap_out)
    # print('bitmap:', bitmap_c)

    # 比较所有分支差之和
    diff = abs(condition[i] - bitmap_c)
    # 比较不一样的分支个数
    nonzero = np.count_nonzero(diff)

    rm_cmd = "rm {}".format(showmap_out)
    _, _ = subprocess_call(rm_cmd)

    return sum(diff), nonzero


def compare(out_path):
    diff_sum = []
    diff_count = []
    for i in range(1280):
        sum, nonzero = compare_single(i, out_path)
        diff_sum.append(sum)
        diff_count.append(nonzero)
    return diff_sum, diff_count


if __name__ == '__main__':
    out_path = output_dir + generate

    # sum, nonzero = compare_single(45, out_path)
    # print('diff_sum:', sum)
    # print('diff_count:', nonzero)

    diff_sum, diff_count = compare(out_path)
    print('diff_sum:', diff_sum)
    print('diff_count:', diff_count)
    np.save('{}/diff_sum.npy'.format(out_path), diff_sum)
    np.save('{}/diff_count.npy'.format(out_path), diff_count)
    
    mean_diff_sum = np.mean(diff_sum)
    mean_diff_count = np.mean(diff_count)

    print('len(diff_sum) len(diff_count): ', len(diff_sum), len(diff_count))
    print("mean_diff_sum:%.2f" % mean_diff_sum)
    print("mean_diff_count:%.2f" % mean_diff_count)
