import os
import sys
import numpy as np
import tensorflow as tf
from sequence.data_process import *
from nn.wd_trainer import WassersteinTrainer
from nn.scale_model import ScaleModel
from config import *
from nn.optimizer import Optimizer

precision = PRECISION
train_data, test_data, max_bytes = load_data(precision)
sys.stdout.write('数据加载完毕，共有{}个batch的训练集！\n'.format(len(train_data)))
sys.stdout.write('max_bytes: {}！\n'.format(max_bytes))


if __name__ == '__main__':
    model = ScaleModel(max_bytes)
    model.build()

    model.generator.summary()
    model.discriminator.summary()
