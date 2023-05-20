import tensorflow as tf
from nn.model import NetModel


class Conv1D_Model(NetModel):
    def generator_simple(self, z_var):
        '''
        args:
            z_var(128)
            c_var(32)+z_var
        return:
            self.s
        describe:
            g_dim: 
            network: 输入层() -> 全连接层(256 or ) -> 一维转置卷积... -> 一维转置卷积[self.n] -> 全连接层(seed_length)[dropout]
        '''
        neurons = 256
        if self.n > neurons:
            z_var = tf.keras.layers.ReLU()(tf.keras.layers.Dense(neurons)(z_var))

        z_var = tf.expand_dims(z_var, -1)

        g_filters = self.g_dim
        while self.n > neurons * 2:
            z_var = tf.keras.layers.ReLU()(tf.keras.layers.BatchNormalization()(tf.keras.layers.Conv1DTranspose(filters=g_filters, kernel_size=4, strides=2, padding="SAME")(z_var)))
            neurons = neurons * 2
            g_filters = max(1, g_filters // 2)
        z_var = tf.keras.layers.Conv1DTranspose(
            filters=1, kernel_size=4, strides=2, padding="SAME")(z_var)

        z_var = tf.keras.layers.Flatten()(z_var)

        z_var = tf.keras.layers.Dropout(0.5)(tf.keras.layers.Dense(self.s, activation='sigmoid')(z_var))
        return z_var

    # stage I discriminator (d)
    def d_encode_sample(self, x_var):
        '''
        args:
            x_var: 4481 * 1
        return:
            out: 128 * ?
        describe:
            D_DIM = 8
        '''
        x_var = tf.keras.layers.ReLU()(tf.keras.layers.Dense(self.n)(x_var))
        x_var = tf.expand_dims(x_var, -1)

        n = self.n
        d = self.d_dim
        ef = self.ef_dim
        while n > ef:
            x_var = tf.keras.layers.Dropout(0.3)(tf.keras.layers.LeakyReLU(alpha=0.2)(tf.keras.layers.Conv1D(filters=d, kernel_size=4, strides=2, padding="SAME")(x_var)))
            n = n // 2
            d = d * 2
        return x_var

    def discriminator_simple(self, var, out_shape):
        '''
        in: (N, 160, ?) / (N, 128, ?)
        out: out_shape = 1 or 4
        '''
        f = var.shape[2]
        var = tf.keras.layers.Dropout(0.3)(tf.keras.layers.LeakyReLU(alpha=0.2)(tf.keras.layers.Conv1D(filters=f, kernel_size=1, strides=1, padding="SAME")(var)))

        n = var.shape[1]
        while n > 16:
            var = tf.keras.layers.Dropout(0.3)(tf.keras.layers.LeakyReLU(alpha=0.2)(tf.keras.layers.Conv1D(filters=f, kernel_size=4, strides=2, padding="SAME")(var)))
            n = n // 2
            f = f // 2
        
        var = tf.keras.layers.Flatten()(var)

        bin_vn = bin(var.shape[-1])
        v_n = pow(2, len(bin_vn)-3)
        while v_n >= 128:
            var = tf.keras.layers.Dropout(0.3)(tf.keras.layers.LeakyReLU(alpha=0.2)(tf.keras.layers.Dense(v_n)(var)))
            v_n = v_n // 2
        
        var = tf.keras.layers.Dense(out_shape)(var)
        return var

    def build_discriminator_wc(self):
        '''
        with condition
        '''
        x_var = tf.keras.Input(shape=self.s)
        c_var = tf.keras.Input(shape=self.c_dim)
        in_var = [c_var, x_var]

        x_code = self.d_encode_sample(x_var)
        f = x_code.shape[-1]
        c_code = self.condition(c_var)
        c_code = tf.expand_dims(c_code, -1)  # shape=(N, 32, 1)
        c_code = tf.tile(c_code, [1, 1, f])  # shape=(N, 32, f)

        x_c_code = tf.concat([x_code, c_code], -2)  # shape=(N, ?, f)

        var = self.discriminator_simple(x_c_code, 1)
        self.discriminator_wi_condition = tf.keras.models.Model(inputs=in_var, outputs=var)