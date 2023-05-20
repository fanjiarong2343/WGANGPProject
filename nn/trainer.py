import os
from config import *
import tensorflow as tf
from datetime import datetime


class Trainer(object):
    def __init__(self, train_data, scale_model, optimizer, lambda_kl=1, lambda_gp=10):
        self.scale_model = scale_model
        self.train_data = train_data
        self.flag = FLAG
        self.max_epoch = MAX_EPOCH

        self.model_path = MODEL_PATH
        self.log_dir = LOG_DIR
        self.train_ratio_I = TRAIN_RATIO_I
        self.train_ratio_II = TRAIN_RATIO_II
        self.lambda_kl = lambda_kl
        self.lambda_gp = lambda_gp

        self.model_type = MODEL_TYPE
        self.decay_epochs = DECAY_EPOCHS
        
        self.optimizer = optimizer
        
        now = datetime.now()
        self.date_time = now.strftime("%m%d%H%M%S")

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.log_name = 'train_' + self.date_time
        self.save_log = os.path.join(self.log_dir, self.log_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log = open(self.save_log, "a")

        self.log.write('flag:{}\n'.format(self.flag))
        self.log.write('max_epoch:{}\n'.format(self.max_epoch))
        self.log.write('train_ratio_I:{}\ntrain_ratio_II:{}\n'.format(self.train_ratio_I, self.train_ratio_II))
        self.log.write('G_LR:{}\nD_LR:{}\n'.format(STAGEI_G_LR, STAGEI_D_LR))
        self.log.write('precision:{}\n'.format(PRECISION))
        self.log.flush()

    def count_klloss(self, mu, log_sigma):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
        loss = tf.reduce_mean(loss)
        return loss
    
    def cross_entropy(self, label, pred, i):
        '''
        In:
        label = [1.0, 0, 0]
        pred = [0.7, 0.1, 0.2]
        i = 0 1 2

        期望 loss --> 0
        '''
        loss = label * tf.math.log(pred)
        loss = tf.slice(loss, [0, i], [loss.shape[0], 1])
        loss = - tf.math.reduce_mean(loss)
        return loss

    def train_step_g_woc(self):
        '''
        return:
            g_loss_fake: 无条件下生成器的损失
        '''
        log_vars = []
        return log_vars

    def train_step_g_wc(self, c):
        '''
        return:
            g_loss_fake: 有条件下生成器的损失
        '''
        log_vars = []
        return log_vars

    def train_step_g_wca(self, c):
        '''
        args:
            c: 条件edges
        return:
            g_loss_fake: 条件增强下生成器的损失
        '''
        log_vars = []
        return log_vars

    def train_step_d_woc(self, lr_seeds):
        '''
        args:
            lr_seeds: 低精度真实种子
        return:
            d_loss_real: 无条件下判别器针对真实种子的损失
            d_loss_fake: 无条件下判别器针对生成种子的损失
            discriminator_loss: 无条件下判别器的损失
        '''
        log_vars = []
        return log_vars


    def train_step_d_wc(self, lr_seeds, c, w_c):
        '''
        args:
            lr_seeds: 低精度真实种子
            c: 条件edges
        return:
            d_loss_real: 有条件下判别器针对真实种子的损失
            d_loss_fake: 有条件下判别器针对生成种子的损失
            discriminator_loss: 有条件下判别器的损失
        '''
        log_vars = []
        return log_vars

    def train_step_d_wca(self, lr_seeds, c, w_c):
        '''
        args:
            lr_seeds: 低精度真实种子
            c: 条件edges
        return:
            d_loss_real: 条件增强下判别器针对真实种子的损失
            d_loss_fake: 条件增强下判别器针对生成种子的损失
            discriminator_loss: 条件增强下判别器的损失
        '''
        log_vars = []
        return log_vars

    def train_step_g(self, lr_seeds, c):
        if self.flag == 'LWOC':
            return self.train_step_g_woc()
        elif self.flag == 'LWCO':
            return self.train_step_g_wc(c)
        else:  # self.flag == 'LWCA':
            return self.train_step_g_wca(c)

    def train_step_d(self, lr_seeds, c, w_c):
        if self.flag == 'LWOC':
            return self.train_step_d_woc(lr_seeds)
        elif self.flag == 'LWCO':
            return self.train_step_d_wc(lr_seeds, c, w_c)
        else:  # self.flag == 'LWCA':
            return self.train_step_d_wca(lr_seeds, c, w_c)

    def train(self):
        cnt = 0
        for epoch in range(self.max_epoch):
            # test_step(args, epoch)
            for real_seed, condition, wrong_condition in self.train_data:
                cnt += 1
                if cnt % self.train_ratio_I != 0:
                    g_log_vars = self.train_step_g(real_seed, condition)
                else:
                    d_log_vars = self.train_step_d(real_seed, condition, wrong_condition)
            
            self.log.write('Epoch: {}\n'.format(epoch + 1))
            for k, v in g_log_vars:
                g_loss_fake = v
                self.log.write('{}: {} '.format(k, v))
            self.log.write('\n')
            for k, v in d_log_vars:
                self.log.write('{}: {} '.format(k, v))
            self.log.write('\n')
            self.log.flush()

            # 学习率更新
            if (epoch + 1) % self.decay_epochs == 0:
                self.optimizer.lr_decay('I')
            
            # if epoch == 40:
            #     min_gloss_fake = g_loss_fake
            #     self.save_model()
            # if epoch > 40 and g_loss_fake < min_gloss_fake:
            #     min_gloss_fake = g_loss_fake
            #     self.save_model()
        self.save_model()

    def save_model(self):
        # print(self.model_path)
        # os.chdir(self.model_path)   # 修改当前工作目录
        submodel_names = ['generator', 'discriminator']
        submodels = [self.scale_model.generator, self.scale_model.discriminator]
        for i in range(2):
            submodel_name = submodel_names[i]
            submodel = submodels[i]
            model_name = '{}_{}_{}_{}'.format(self.date_time, self.model_type, self.flag, submodel_name)
            print("model_name:", model_name)
            model_path = self.model_path + '/{}'.format(model_name)
            print("model_path:", model_path)
            submodel.save_weights(model_path)
