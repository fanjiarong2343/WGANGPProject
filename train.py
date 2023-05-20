import os
import sys
import numpy as np
import tensorflow as tf
from sequence.data_process import load_data
from nn.wd_trainer import WassersteinTrainer
from nn.scale_model import ScaleModel
from config import *
from nn.optimizer import Optimizer


precision = PRECISION
train_data, test_data, max_bytes = load_data(precision)
sys.stdout.write('数据加载完毕，共有{}个batch的训练集！\n'.format(len(train_data)))
sys.stdout.write('max_bytes: {}！\n'.format(max_bytes))


if __name__ == '__main__':
    print("请核对以下信息是否正确：\nFLAG:{}\nPROGRAM:{}\nPRECISION:{}\nMAX_COUNT:{}\nC_DIM:{}\nMAX_EPOCH:{}\nG_LR:{}\nD_LR:{}\n".format(FLAG, PROGRAM, PRECISION, MAX_COUNT, C_DIM, MAX_EPOCH, STAGEI_G_LR, STAGEI_D_LR))
    
    s = input()
    if s != 'Y':
        exit()
    model = ScaleModel(max_bytes)
    model.build()

    optimizer = Optimizer()
    optimizer.init_opt()

    trainer = WassersteinTrainer(train_data, model, optimizer)

    trainer.train()
