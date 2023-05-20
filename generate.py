import os
import sys
import numpy as np
import tensorflow as tf
from sequence.data_process import *
from nn.wd_trainer import WassersteinTrainer
from nn.scale_model import ScaleModel
from config import *
import random
from datetime import datetime
from nn.optimizer import Optimizer
from nn.load_model import load_model
from sequence.utils import save_vector


precision = PRECISION
train_data, test_data, max_bytes = load_data(precision)
sys.stdout.write('加载数据完毕，共有{}个batch的训练集！\n'.format(len(train_data)))

now = datetime.now()
date_time = now.strftime("%m%d%H%M%S")

output_dir = OUTPUT_DIR


def generate_c(scale_model, r_b=10, o_b=10, c_b=10):
    file_path = output_dir + '/generated_' + date_time + '_' + precision + 'bits'
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    # 随机分支
    generate_random = None
    c_randoms = None
    for i in range(r_b):
        c_random = tf.random.uniform(shape=(BATCH_SIZE, C_DIM))
        # c_random = tf.random.uniform(shape=(BATCH_SIZE, C_DIM), minval=0.5, maxval=1)
        generate_var = scale_model.model.generate_wc(c_random)
        if i == 0:
            generate_random = generate_var
            c_randoms = c_random
        else:
            generate_random = tf.concat([generate_random, generate_var], 0)
            c_randoms = tf.concat([c_randoms, c_random], 0)
    np.save(os.path.join(file_path, 'c_random.npy'), c_randoms.numpy())
    
    # 训练集的分支
    c_original = None
    generate_original = None
    o = 0
    for seeds, condition, wrong_condition in train_data:
        if o >= o_b:
            break
        generate_var = scale_model.model.generate_wc(condition)
        if o == 0:
            generate_original = generate_var
            c_original = condition
        else:
            generate_original = tf.concat([generate_original, generate_var], 0)
            c_original = tf.concat([c_original, condition], 0)
        o += 1
    np.save(os.path.join(file_path, 'c_original.npy'), c_original.numpy())

    # 拼接分支
    c_concats = None
    generate_concat = None
    c = 0
    for seeds, condition, wrong_condition in train_data:
        if c >= c_b:
            break
        index_list = []
        for i in range(3):
            index_list.append(random.randint(1, C_DIM-1))
        index_list = list(set(index_list))  # 先将列表转化为set，再转化为list就可以实现去重操作
        index_list.sort()

        i_a = index_list[0]
        i_b = index_list[1]
        i_c = index_list[2]

        # c_concat = tf.concat([condition[:, :i_a], wrong_condition[:, i_a:]], 1)
        c_concat = tf.concat([condition[:, :i_a], wrong_condition[:, i_a:i_b], condition[:, i_b:]], 1)
        # c_concat = tf.concat([condition[:, :i_a], wrong_condition[:, i_a:i_b], condition[:, i_b:i_c], wrong_condition[:, i_c:]], 1)

        generate_var = scale_model.model.generate_wc(c_concat)
        if c == 0:
            generate_concat = generate_var
            c_concats = c_concat
        else:
            generate_concat = tf.concat([generate_concat, generate_var], 0)
            c_concats = tf.concat([c_concats, c_concat], 0)
        c += 1
    np.save(os.path.join(file_path, 'c_concat.npy'), c_concats.numpy())

    generate_random = tf.reshape(generate_random, (generate_random.shape[0], -1))
    generate_original = tf.reshape(generate_original, (generate_original.shape[0], -1))
    generate_concat = tf.reshape(generate_concat, (generate_concat.shape[0], -1))
    
    save_vector(arr=generate_random, dir_name='all/random', dir_path=file_path, label=precision+'bits')
    save_vector(arr=generate_original, dir_name='all/original', dir_path=file_path, label=precision+'bits')
    save_vector(arr=generate_concat, dir_name='all/concat', dir_path=file_path, label=precision+'bits')


def generate(scale_model, flag, b):
    generate_seeds = None
    if flag == 'LWOC':
        for i in range(b):
            generate_var = scale_model.model.generate_woc()
            if generate_seeds is None:
                generate_seeds = generate_var
            else:
                generate_seeds = tf.concat([generate_seeds, generate_var], 0)

        generate_seeds = tf.reshape(generate_seeds, (generate_seeds.shape[0], -1))
        filename = 'generated_' + date_time + '_' + precision + 'bits'
        save_vector(arr=generate_seeds, dir_name=filename, dir_path=output_dir, label=precision+'bits')

    elif flag == 'LWCO':
        # 以分支作为条件生成种子的代码（ 拼接分支信息 / 随机生成一些 / 训练集的分支信息）
        generate_c(scale_model)

    elif flag == 'LWCA':
        for seeds, condition, wrong_condition in train_data:
            np.save("condition.npy", condition.numpy())
            generate_var, _ = scale_model.model.generate_wca(condition)
            if generate_seeds is None:
                generate_seeds = generate_var
            else:
                generate_seeds = tf.concat([generate_seeds, generate_var], 0)
            break

    else:
        pass


if __name__ == '__main__':
    flag = FLAG
    scale_model = ScaleModel(max_bytes)
    scale_model.build()
    model_path = MODEL_PATH
    model_type = MODEL_TYPE
    scale_model = load_model(scale_model, model_path, flag, model_type)

    batchs = 10
    generate(scale_model, flag, batchs)
