import tensorflow as tf


def out_neurons(seed_length, is_onedim=True):
    '''
    args:
        seed_length: 输入测试用例的长度
        is_onedim: 网络模型的输入是一维数据
    return:
        <=seed_length 的2的n次方的最大值
    '''
    # https://blog.csdn.net/bitzhidu/article/details/106386742
    bin_sl = bin(seed_length)
    neurons = pow(2, len(bin_sl)-3)

    if is_onedim:  # 一维
        # return neurons
        return neurons // 2  # readelf
    else:
        if (len(bin_sl)-3) % 2 == 0:
            return neurons // 2
        else:
            return neurons


class NetModel(tf.keras.Model):
    def __init__(self, batch_size, seed_length, c_dim, z_dim, ef_dim, g_dim, d_dim, precision, is_onedim=True):
        super(NetModel, self).__init__()
        self.batch_size = batch_size

        self.c_dim = c_dim
        self.z_dim = z_dim
        self.ef_dim = ef_dim
        self.g_dim = g_dim
        self.d_dim = d_dim

        if precision == '16':
            self.n = out_neurons(seed_length, is_onedim)//2
            self.s = seed_length//2
        elif precision == '8':
            self.n = out_neurons(seed_length, is_onedim)
            self.s = seed_length
        elif precision == '4':
            self.n = out_neurons(seed_length, is_onedim)*2
            self.s = seed_length*2
        elif precision == '2':
            self.n = out_neurons(seed_length, is_onedim)*4
            self.s = seed_length*4
        elif precision == '1':
            self.n = out_neurons(seed_length, is_onedim)*8
            self.s = seed_length*8
        else:
            print("ERROR!")

    def encode_condition(self, c_var):
        '''
        args:
            c_var: 条件变量
        return:
            
        '''
        bin_cd = bin(self.c_dim)
        n = pow(2, len(bin_cd)-3)

        # c_var = tf.keras.layers.Flatten()(c_var)
        while n >= self.ef_dim * 2:
            c_var = tf.keras.layers.LeakyReLU(alpha=0.2)(tf.keras.layers.Dense(n)(c_var))
            n = n // 2
        
        return c_var

    def condition(self, c_var):
        '''
        args:
            c_var
        return:
            encode_condition
        '''
        c_var = self.encode_condition(c_var)
        conditions = tf.keras.layers.LeakyReLU(alpha=0.2)(tf.keras.layers.Dense(self.ef_dim)(c_var))
        return conditions

    def condition_augment(self, c_var):
        '''
        args:
            c_var
        return:
            mean: 条件服从的分布的均值
            log_sigma: 条件服从的分布的方差
        '''
        conditions = self.encode_condition(c_var)
        mean = conditions[:, :self.ef_dim]
        log_sigma = conditions[:, self.ef_dim:]
        return [mean, log_sigma]

    def reparameterize(self, tuple_x):
        mean, logvar = tuple_x
        eps = tf.random.normal(shape=[self.ef_dim])
        return eps * tf.exp(logvar * .5) + mean

    # generator (g)
    def generator_simple(self, z_var):
        return z_var

    def build_generator_woc(self):
        '''
        without condition
        '''
        z_var = tf.keras.Input(shape=(self.z_dim))
        generate_var = self.generator_simple(z_var)
        self.generator_wo_condition = tf.keras.models.Model(inputs=z_var, outputs=generate_var)

    def generate_woc(self):
        '''
        without condition
        '''
        z_var = tf.random.normal(shape=(self.batch_size, self.z_dim))
        generate_var = self.generator_wo_condition(z_var)
        return generate_var

    def build_generator_wc(self):
        '''
        with condition edges
        '''
        c_code = tf.keras.Input(shape=(self.c_dim))
        z_var = tf.keras.Input(shape=(self.z_dim))
        
        c_var = self.condition(c_code)
        in_var = tf.concat([c_var, z_var], -1)
        generate_var = self.generator_simple(in_var)
        self.generator_wi_condition = tf.keras.models.Model(inputs=[c_code, z_var], outputs=generate_var)

    def generate_wc(self, c_var):
        '''
        with condition
        '''
        z_var = tf.random.normal(shape=(self.batch_size, self.z_dim))
        generate_var = self.generator_wi_condition([c_var, z_var])
        return generate_var

    def build_generator_wca(self):
        '''
        with condition augment
        In: (N, 1042), (N, 100)
        '''
        c_var = tf.keras.Input(shape=(self.c_dim))
        tuple_x = self.condition_augment(c_var)
        c_eps = self.reparameterize(tuple_x)  # (N, 100)

        z_var = tf.keras.Input(shape=(self.z_dim))  # (N, 100)
        cz_var = tf.concat([c_eps, z_var], 1)  # shape=(N, 200)
        generate_var = self.generator_simple(cz_var)
        self.generator_wi_conaugment = tf.keras.models.Model(inputs=[c_var, z_var], outputs=[generate_var, tuple_x])

    def generate_wca(self, c_var):
        '''
        with condition augment
        '''
        z_var = tf.random.normal(shape=(self.batch_size, self.z_dim))
        generate_var, tuple_x = self.generator_wi_conaugment([c_var, z_var])
        return generate_var, tuple_x

    # discriminator (d)
    def d_encode_sample(self, x_var):
        '''
        args:
            x_var: 
        return:
        
        describe:

        '''
        return x_var

    def discriminator_simple(self, var, out_shape):
        '''
        args:
            var:
            out_shape:
        return:
        
        describe:

        '''
        return var

    def build_discriminator_woc(self):
        '''
        without condition
        '''
        x_var = tf.keras.Input(shape=self.s)
        x_code = self.d_encode_sample(x_var)
        var = self.discriminator_simple(x_code, 1)
        self.discriminator_wo_condition = tf.keras.models.Model(inputs=x_var, outputs=var)

    def discriminate_woc(self, x_var):
        '''
        without condition
        '''
        discriminate_var = self.discriminator_wo_condition(x_var)
        return discriminate_var

    def build_discriminator_wc(self):
        '''
        with condition
        In:
        (N, 1042), [N, 128, 128, 1]
        shape=(N, 100), [N, 8, 8, 64]
        '''
        x_var = tf.keras.Input(shape=self.s)
        c_var = tf.keras.Input(shape=self.c_dim)
        in_var = [c_var, x_var]
        out_var = []
        self.discriminator_wi_condition = tf.keras.models.Model(inputs=in_var, outputs=out_var)

    def discriminate_wc(self, in_var):
        '''
        with condition
        '''
        discriminate_var = self.discriminator_wi_condition(in_var)
        return discriminate_var
