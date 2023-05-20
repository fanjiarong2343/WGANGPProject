import numpy as np
import tensorflow as tf
from sequence.DataSequence import DataSequence
from nn.conv1d import Conv1D_Model
from config import *


class ScaleModel(tf.keras.Model):
    def __init__(self, seed_length):
        super(ScaleModel, self).__init__()
        self.model_type = MODEL_TYPE
        g_dim = G_DIM
        z_dim = Z_DIM
        c_dim = C_DIM
        ef_dim = EMBEDDING_DIM
        batch_size = BATCH_SIZE
        d_dim = D_DIM
        precision = PRECISION
        if self.model_type == "conv1d":
            self.model = Conv1D_Model(batch_size, seed_length, c_dim, z_dim, ef_dim, g_dim, d_dim, precision)
        else:
            pass
        self.flag = FLAG

    def build(self):
        if self.flag == 'LWOC':
            self.model.build_generator_woc()
            self.model.build_discriminator_woc()
            self.generator = self.model.generator_wo_condition
            self.discriminator = self.model.discriminator_wo_condition

        elif self.flag == 'LWCO':
            self.model.build_generator_wc()
            self.model.build_discriminator_wc()
            self.generator = self.model.generator_wi_condition
            self.discriminator = self.model.discriminator_wi_condition

        elif self.flag == 'LWCA':
            self.model.build_generator_wca()
            self.model.build_discriminator_wc()
            self.generator = self.model.generator_wi_conaugment
            self.discriminator = self.model.discriminator_wi_condition

        else:
            pass

