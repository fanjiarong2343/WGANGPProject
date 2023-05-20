from config import *
import tensorflow as tf


class Optimizer(object):
    def __init__(self):
        self.stageI_g_lr = STAGEI_G_LR
        self.stageI_d_lr = STAGEI_D_LR
        self.stageII_gI_lr = STAGEII_GI_LR
        self.stageII_gII_lr = STAGEII_GII_LR
        self.stageII_dII_lr = STAGEII_DII_LR
        self.decay = DECAY

        self.opt_type = OPT_TYPE

    def init_opt(self):
        '''
        In: lr
        Out: opt
        '''
        if self.opt_type == "adam":
            # 第一阶段
            self.stageI_g_opt = tf.keras.optimizers.Adam(self.stageI_g_lr, beta_1=0.9, beta_2=0.99)
            self.stageI_d_opt = tf.keras.optimizers.Adam(self.stageI_d_lr, beta_1=0.9, beta_2=0.99)
            # 第二阶段
            self.stageII_gI_opt = tf.keras.optimizers.Adam(self.stageII_gI_lr, beta_1=0.9, beta_2=0.99)
            self.stageII_gII_opt = tf.keras.optimizers.Adam(self.stageII_gII_lr, beta_1=0.9, beta_2=0.99)
            self.stageII_dII_opt = tf.keras.optimizers.Adam(self.stageII_dII_lr, beta_1=0.9, beta_2=0.99)

        elif self.opt_type == "rmsprop":
            # 第一阶段
            self.stageI_g_opt = tf.keras.optimizers.RMSprop(self.stageI_g_lr)
            self.stageI_d_opt = tf.keras.optimizers.RMSprop(self.stageI_d_lr)
            # 第二阶段
            self.stageII_gI_opt = tf.keras.optimizers.RMSprop(self.stageII_gI_lr)
            self.stageII_gII_opt = tf.keras.optimizers.RMSprop(self.stageII_gII_lr)
            self.stageII_dII_opt = tf.keras.optimizers.RMSprop(self.stageII_dII_lr)

        elif self.opt_type == "sgd":
            # 第一阶段
            self.stageI_g_opt = tf.keras.optimizers.SGD(self.stageI_g_lr)
            self.stageI_d_opt = tf.keras.optimizers.SGD(self.stageI_d_lr)
            # 第二阶段
            self.stageII_gI_opt = tf.keras.optimizers.SGD(self.stageII_gI_lr)
            self.stageII_gII_opt = tf.keras.optimizers.SGD(self.stageII_gII_lr)
            self.stageII_dII_opt = tf.keras.optimizers.SGD(self.stageII_dII_lr)

        else:
            pass

    def lr_decay(self, stage='I'):
        # 学习率更新
        if stage == 'I':
            self.stageI_g_opt.learning_rate = self.stageI_g_opt.learning_rate * self.decay
            self.stageI_d_opt.learning_rate = self.stageI_d_opt.learning_rate * self.decay
        elif stage == 'II':
            self.stageII_gI_opt.learning_rate = self.stageII_gI_opt.learning_rate * self.decay
            self.stageII_gII_opt.learning_rate = self.stageII_gII_opt.learning_rate * self.decay
            self.stageII_dII_opt.learning_rate = self.stageII_dII_opt.learning_rate * self.decay
        else:
            pass
