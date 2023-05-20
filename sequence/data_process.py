import os
import numpy as np
import tensorflow as tf
from sequence.DataSequence import DataSequence
from config import *
from revise_config import *


flag = FLAG
if flag == 'LWOC':
    input_dir = FORMAT_DIR
else:
    input_dir = EDGE_DIR
batch_size = BATCH_SIZE
showmap = SHOWMAP
output_dir = OUTPUT_DIR
bitmap_dir = BITMAP_DIR + INDEX
program_path = PROGRAM_PATH
parameter = PARAMETER
nparray_dir = OUTPUT_DIR


def file_nparray():
    dataSequence = DataSequence(input_dir, batch_size, showmap, bitmap_dir, program_path, parameter)

    dataSequence.aflshowmap()
    dataSequence.bitmap()
    revise(['', 'MAX_COUNT', 'int', dataSequence.max_count])
    revise(['', 'C_DIM', 'int', dataSequence.edges_dim])
    dataSequence.split_data()

    # 保存文件(执行一次即可)
    np.save("{}/train_16bits_{}.npy".format(nparray_dir, INDEX), dataSequence.train_16bits_seed)
    np.save("{}/train_8bits_{}.npy".format(nparray_dir, INDEX), dataSequence.train_8bits_seed)
    np.save("{}/train_4bits_{}.npy".format(nparray_dir, INDEX), dataSequence.train_4bits_seed)
    np.save("{}/train_2bits_{}.npy".format(nparray_dir, INDEX), dataSequence.train_2bits_seed)
    np.save("{}/train_1bits_{}.npy".format(nparray_dir, INDEX), dataSequence.train_1bits_seed)
    np.save("{}/train_edge_{}.npy".format(nparray_dir, INDEX), dataSequence.train_edge)
    np.save("{}/test_16bits_{}.npy".format(nparray_dir, INDEX), dataSequence.test_16bits_seed)
    np.save("{}/test_8bits_{}.npy".format(nparray_dir, INDEX), dataSequence.test_8bits_seed)
    np.save("{}/test_4bits_{}.npy".format(nparray_dir, INDEX), dataSequence.test_4bits_seed)
    np.save("{}/test_2bits_{}.npy".format(nparray_dir, INDEX), dataSequence.test_2bits_seed)
    np.save("{}/test_1bits_{}.npy".format(nparray_dir, INDEX), dataSequence.test_1bits_seed)
    np.save("{}/test_edge_{}.npy".format(nparray_dir, INDEX), dataSequence.test_edge)


def load_data(precision):
    dataSequence = DataSequence(input_dir, batch_size, showmap, bitmap_dir, program_path, parameter)
    max_bytes = dataSequence.max_bytes

    # 读取文件
    dataSequence.train_16bits_seed = np.load("{}/train_16bits_{}.npy".format(nparray_dir, INDEX))
    dataSequence.train_8bits_seed = np.load("{}/train_8bits_{}.npy".format(nparray_dir, INDEX))
    dataSequence.train_4bits_seed = np.load("{}/train_4bits_{}.npy".format(nparray_dir, INDEX))
    dataSequence.train_2bits_seed = np.load("{}/train_2bits_{}.npy".format(nparray_dir, INDEX))
    dataSequence.train_1bits_seed = np.load("{}/train_1bits_{}.npy".format(nparray_dir, INDEX))
    dataSequence.train_edge = np.load("{}/train_edge_{}.npy".format(nparray_dir, INDEX))
    dataSequence.test_16bits_seed = np.load("{}/test_16bits_{}.npy".format(nparray_dir, INDEX))
    dataSequence.test_8bits_seed = np.load("{}/test_8bits_{}.npy".format(nparray_dir, INDEX))
    dataSequence.test_4bits_seed = np.load("{}/test_4bits_{}.npy".format(nparray_dir, INDEX))
    dataSequence.test_2bits_seed = np.load("{}/test_2bits_{}.npy".format(nparray_dir, INDEX))
    dataSequence.test_1bits_seed = np.load("{}/test_1bits_{}.npy".format(nparray_dir, INDEX))
    dataSequence.test_edge = np.load("{}/test_edge_{}.npy".format(nparray_dir, INDEX))

    dataSequence.shuffle(precision)
    return dataSequence.train_data, dataSequence.test_data, max_bytes


