import os
import numpy as np
from glob import glob
import tensorflow as tf
from collections import Counter
from sequence.utils import *
from config import *


class DataSequence():
    def __init__(self, input_file_dir: str, batch_size, showmap, bitmap_dir, program_path, parameter, split_ratio=0.8):
        self.input_file_dir = input_file_dir
        self.input_files: List[str] = glob("%s/*" % (self.input_file_dir))
        self.total_files = len(self.input_files)
        self.split_ratio = split_ratio
        self.batch_size: int = batch_size
        self.max_bytes: int = get_max_file_bytes(self.input_file_dir)

        self.showmap = showmap
        self.bitmap_dir = bitmap_dir
        self.program_path = program_path
        self.parameter = parameter

    def aflshowmap(self):
        if not os.path.isdir(self.bitmap_dir):
            os.mkdir(self.bitmap_dir)
        cmd = "export AFL_MAP_SIZE=102400K; {} -i {} -o {} -- {} {}".format(self.showmap, self.input_file_dir, self.bitmap_dir, self.program_path, self.parameter)
        # print("afl-showmap:", cmd)
        _, _ = subprocess_call(cmd)
        return 0
    
    def bitmap(self):
        raw_bitmap = {}
        tmp_cnt = []
        max_count = 0
        for file in self.input_files:
            tmp_list = []

            file_name = file.split('/')[-1]
            file_path = os.path.join(self.bitmap_dir, file_name)
            content = open(file_path, "r", encoding='ISO-8859-1')  # 打开文件
            for line in content.readlines():
                line = line.strip()  # 去掉每行头尾空白
                edge = line.split(':')[0]
                count = int(line.split(':')[1])
                if count > max_count:
                    max_count = count
                tmp_cnt.append(edge)
                tmp_list.append((edge, count))
            content.close()  # 关闭文件
            raw_bitmap[file] = tmp_list
        
        self.max_count = max_count
        counter = Counter(tmp_cnt).most_common()

        # save bitmaps to individual numpy label
        label = [e[0] for e in counter]
        if FLAG != "LWOC":
            np_label = np.array(label)
            np.save("{}/dictkey_{}.npy".format(OUTPUT_DIR, INDEX), np_label)
        
        bitmap = np.zeros((len(self.input_files), len(label)))
        for idx, i in enumerate(self.input_files):
            tmp = raw_bitmap[i]
            for j in tmp:
                if j[0] in label:
                    bitmap[idx][label.index(j[0])] = j[1]
        self.edges_dim = len(label)
        self.bitmap = bitmap / max_count  # 归一化

    def split_data(self):
        self.train_16bits_seed = []
        self.train_8bits_seed = []
        self.train_4bits_seed = []
        self.train_2bits_seed = []
        self.train_1bits_seed = []
        self.train_edge = []
        self.test_16bits_seed = []
        self.test_8bits_seed = []
        self.test_4bits_seed = []
        self.test_2bits_seed = []
        self.test_1bits_seed = []
        self.test_edge = []

        for idx, input_file in enumerate(self.input_files):
            # lr_seed hr_seed 读取并转为 numpy 矩阵
            _16bits_seed, _8bits_seed, _4bits_seed, _2bits_seed, _1bits_seed = vectorize_file(input_file, self.max_bytes)
            
            if idx <= self.total_files * self.split_ratio:
                self.train_16bits_seed.append(_16bits_seed)
                self.train_8bits_seed.append(_8bits_seed)
                self.train_4bits_seed.append(_4bits_seed)
                self.train_2bits_seed.append(_2bits_seed)
                self.train_1bits_seed.append(_1bits_seed)
                self.train_edge.append(self.bitmap[idx])
            else:
                self.test_16bits_seed.append(_16bits_seed)
                self.test_8bits_seed.append(_8bits_seed)
                self.test_4bits_seed.append(_4bits_seed)
                self.test_2bits_seed.append(_2bits_seed)
                self.test_1bits_seed.append(_1bits_seed)
                self.test_edge.append(self.bitmap[idx])

        self.train_16bits_seed = np.array(self.train_16bits_seed)
        self.train_8bits_seed = np.array(self.train_8bits_seed)
        self.train_4bits_seed = np.array(self.train_4bits_seed)
        self.train_2bits_seed = np.array(self.train_2bits_seed)
        self.train_1bits_seed = np.array(self.train_1bits_seed)
        self.train_edge = np.array(self.train_edge)
        self.test_16bits_seed = np.array(self.test_16bits_seed)
        self.test_8bits_seed = np.array(self.test_8bits_seed)
        self.test_4bits_seed = np.array(self.test_4bits_seed)
        self.test_2bits_seed = np.array(self.test_2bits_seed)
        self.test_1bits_seed = np.array(self.test_1bits_seed)
        self.test_edge = np.array(self.test_edge)

    def shuffle(self, p):
        TRAIN_BUF = 600
        TEST_BUF = 100
        # 训练集
        train_16bits_seed = tf.data.Dataset.from_tensor_slices(self.train_16bits_seed).batch(self.batch_size, drop_remainder=True)
        train_8bits_seed = tf.data.Dataset.from_tensor_slices(self.train_8bits_seed).batch(self.batch_size, drop_remainder=True)
        train_4bits_seed = tf.data.Dataset.from_tensor_slices(self.train_4bits_seed).batch(self.batch_size, drop_remainder=True)
        train_2bits_seed = tf.data.Dataset.from_tensor_slices(self.train_2bits_seed).batch(self.batch_size, drop_remainder=True)
        train_1bits_seed = tf.data.Dataset.from_tensor_slices(self.train_1bits_seed).batch(self.batch_size, drop_remainder=True)
        train_edge = tf.data.Dataset.from_tensor_slices(self.train_edge)
        train_wrong_edges = train_edge.shuffle(TRAIN_BUF)
        train_edge = train_edge.batch(self.batch_size, drop_remainder=True)
        train_wrong_edges = train_wrong_edges.batch(self.batch_size, drop_remainder=True)
        
        # 测试集
        test_16bits_seed = tf.data.Dataset.from_tensor_slices(self.test_16bits_seed).batch(self.batch_size, drop_remainder=True)
        test_8bits_seed = tf.data.Dataset.from_tensor_slices(self.test_8bits_seed).batch(self.batch_size, drop_remainder=True)
        test_4bits_seed = tf.data.Dataset.from_tensor_slices(self.test_4bits_seed).batch(self.batch_size, drop_remainder=True)
        test_2bits_seed = tf.data.Dataset.from_tensor_slices(self.test_2bits_seed).batch(self.batch_size, drop_remainder=True)
        test_1bits_seed = tf.data.Dataset.from_tensor_slices(self.test_1bits_seed).batch(self.batch_size, drop_remainder=True)
        test_edge = tf.data.Dataset.from_tensor_slices(self.test_edge)
        test_wrong_edges = test_edge.shuffle(TEST_BUF)
        test_edge = test_edge.batch(self.batch_size, drop_remainder=True)
        test_wrong_edges = test_wrong_edges.batch(self.batch_size, drop_remainder=True)

        if p == '16':
            self.train_data = tf.data.Dataset.zip((train_16bits_seed, train_edge, train_wrong_edges)).shuffle(TRAIN_BUF)
            self.test_data = tf.data.Dataset.zip((test_16bits_seed, test_edge, test_wrong_edges)).shuffle(TEST_BUF)
        elif p == '8':
            self.train_data = tf.data.Dataset.zip((train_8bits_seed, train_edge, train_wrong_edges)).shuffle(TRAIN_BUF)
            self.test_data = tf.data.Dataset.zip((test_8bits_seed, test_edge, test_wrong_edges)).shuffle(TEST_BUF)
        elif p == '4':
            self.train_data = tf.data.Dataset.zip((train_4bits_seed, train_edge, train_wrong_edges)).shuffle(TRAIN_BUF)
            self.test_data = tf.data.Dataset.zip((test_4bits_seed, test_edge, test_wrong_edges)).shuffle(TEST_BUF)
        elif p == '2':
            self.train_data = tf.data.Dataset.zip((train_2bits_seed, train_edge, train_wrong_edges)).shuffle(TRAIN_BUF)
            self.test_data = tf.data.Dataset.zip((test_2bits_seed, test_edge, test_wrong_edges)).shuffle(TEST_BUF)
        else:
            self.train_data = tf.data.Dataset.zip((train_1bits_seed, train_edge, train_wrong_edges)).shuffle(TRAIN_BUF)
            self.test_data = tf.data.Dataset.zip((test_1bits_seed, test_edge, test_wrong_edges)).shuffle(TEST_BUF)
