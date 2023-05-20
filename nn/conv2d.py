import tensorflow as tf
from nn.model import NetModel


class Conv2D_Model(NetModel):


class Generator(tf.keras.Model):
    def __init__(self, batch_size, g_dim, z_dim, c_dim, ef_dim, generate_shape_I, generate_shape_II):
        super(Generator, self).__init__()
        self.batch_size = batch_size
        self.g_dim = g_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.ef_dim = ef_dim

        self.generate_shape_I = generate_shape_I
        self.generate_shape_II = generate_shape_II
        self.s = self.generate_shape_I[0]
        self.s2, self.s4, self.s8, self.s16 = int(
            self.s / 2), int(self.s / 4), int(self.s / 8), int(self.s / 16)

    def condition(self, c_var):
        self.flatten = tf.keras.layers.Flatten()
        c_var = self.flatten(c_var)
        self.fc = tf.keras.layers.Dense(self.ef_dim * 2)
        c_var = self.fc(c_var)
        self.ac = tf.keras.layers.LeakyReLU(alpha=0.2)
        conditions = self.ac(c_var)
        mean = conditions[:, :self.ef_dim]
        log_sigma = conditions[:, self.ef_dim:]
        return [mean, log_sigma]

    def reparameterize(self, tuple_x):
        mean, logvar = tuple_x
        eps = tf.random.normal(shape=[self.ef_dim])
        return eps * tf.exp(logvar * .5) + mean

    def generator_simple(self, z_var):
        self.flatten = tf.keras.layers.Flatten()
        z_var = self.flatten(z_var)
        self.fc = tf.keras.layers.Dense(self.s16 * self.s16 * self.g_dim * 4)
        z_var = self.fc(z_var)
        self.reshape = tf.keras.layers.Reshape(
            target_shape=(self.s16, self.s16, self.g_dim * 4))
        z_var = self.reshape(z_var)

        self.convtran1 = tf.keras.layers.Conv2DTranspose(
            filters=self.g_dim * 4, kernel_size=3, strides=(2, 2), padding="SAME")
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        z_var = self.relu1(self.batch_norm1(self.convtran1(z_var)))

        self.convtran2 = tf.keras.layers.Conv2DTranspose(
            filters=self.g_dim * 2, kernel_size=3, strides=(2, 2), padding="SAME")
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()
        z_var = self.relu2(self.batch_norm2(self.convtran2(z_var)))

        self.convtran3 = tf.keras.layers.Conv2DTranspose(
            filters=self.g_dim, kernel_size=3, strides=(2, 2), padding="SAME")
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()
        z_var = self.relu3(self.batch_norm3(self.convtran3(z_var)))

        self.convtran4 = tf.keras.layers.Conv2DTranspose(
            filters=1, kernel_size=3, strides=(2, 2), padding="SAME", activation='sigmoid')
        z_var = self.convtran4(z_var)

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
        with condition(refuse/edges)
        '''
        c_var = tf.keras.Input(shape=(self.c_dim))
        z_var = tf.keras.Input(shape=(self.z_dim))
        cz_var = tf.concat([c_var, z_var])
        generate_var = self.generator_simple(cz_var)
        self.generator_wi_condition = tf.keras.models.Model(inputs=cz_var, outputs=generate_var)

    def generate_wc(self, c_var):
        '''
        with condition
        '''
        z_var = tf.random.normal(shape=(self.batch_size, self.z_dim))
        cz_var = tf.concat([c_var, z_var])
        generate_var = self.generator_wi_condition(cz_var)
        return generate_var

    def build_generator_wca(self):
        '''
        with condition augment
        In: (N, 1042), (N, 100)
        '''
        c_var = tf.keras.Input(shape=(self.c_dim))
        tuple_x = self.condition(c_var)
        c_eps = self.reparameterize(tuple_x)  # (N, 100)

        z_var = tf.keras.Input(shape=(self.z_dim))  # (N, 100)
        cz_var = tf.concat([c_eps, z_var], 1)  # shape=(N, 200)
        generate_var = self.generator_simple(cz_var)
        self.generator_wi_conaugment = tf.keras.models.Model(inputs=[c_var, z_var], outputs=[generate_var, tuple_x])

    def generate_wca(self, c_var):
        '''
        with condition augment
        '''
        # z_var = tf.random.normal(shape=(self.batch_size, self.z_dim))
        z_var = tf.random.normal(shape=(c_var.shape[0], self.z_dim))
        generate_var, tuple_x = self.generator_wi_conaugment([c_var, z_var])
        return generate_var, tuple_x

    # stage II generator (hr_g)
    def hr_g_encode_sample(self, x_var):
        '''
        in: s * s * 1
        out: s4 * s4 * g_dim * 4
        '''
        self.conv1 = tf.keras.layers.Conv2D(
            filters=self.g_dim, kernel_size=3, strides=(1, 1), padding="SAME")
        self.ac1 = tf.keras.layers.ReLU()
        x_var = self.ac1(self.conv1(x_var))

        self.conv2 = tf.keras.layers.Conv2D(
            filters=self.g_dim * 2, kernel_size=3, strides=(2, 2), padding="SAME")
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.ac2 = tf.keras.layers.ReLU()
        x_var = self.ac2(self.batch_norm2(self.conv2(x_var)))

        self.conv3 = tf.keras.layers.Conv2D(
            filters=self.g_dim * 4, kernel_size=3, strides=(2, 2), padding="SAME")
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.ac3 = tf.keras.layers.ReLU()
        x_var = self.ac3(self.batch_norm3(self.conv3(x_var)))
        return x_var

    def hr_g_joint(self, x_c_code):
        ''''
        in: s4 * s4 * g_dim * 4 / s4 * s4 * (ef_dim + g_dim * 4)
        out: s4 * s4 * g_dim * 4
        '''
        self.conv = tf.keras.layers.Conv2D(
            filters=self.g_dim * 4, kernel_size=3, strides=(1, 1), padding="SAME")
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.ac = tf.keras.layers.ReLU()
        xc_var = self.ac(self.batch_norm(self.conv(x_c_code)))
        return xc_var
    
    def residual_block(self, x_c_code):
        '''
        In: s4 * s4 * gf_dim * 4
        Out: s4 * s4 * gf_dim * 4
        '''
        self.convtran1 = tf.keras.layers.Conv2DTranspose(
            filters=self.g_dim * 4, kernel_size=3, strides=(1, 1), padding="SAME")
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        xc_var = self.relu1(self.batch_norm1(self.convtran1(x_c_code)))

        self.convtran2 = tf.keras.layers.Conv2DTranspose(
            filters=self.g_dim * 4, kernel_size=3, strides=(1, 1), padding="SAME")
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        xc_var_1 = self.batch_norm2(self.convtran2(xc_var))

        xc_var_10 = tf.add(x_c_code, xc_var_1)
        self.relu = tf.keras.layers.ReLU()
        xc_var_10 = self.relu(xc_var_10)
        return xc_var_10

    def hr_generator_simple(self, z_var):
        '''
        in: s4 * s4 * gf_dim*4
        out: 2s * 2s * 1
        tf.image.resize_nearest_neighbor() 使用最近邻插值调整images的size.
        '''
        self.convtran1 = tf.keras.layers.Conv2DTranspose(
            filters=self.g_dim * 2, kernel_size=3, strides=(2, 2), padding="SAME")
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        z_var = self.relu1(self.batch_norm1(self.convtran1(z_var)))

        self.convtran2 = tf.keras.layers.Conv2DTranspose(
            filters=self.g_dim, kernel_size=3, strides=(2, 2), padding="SAME")
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()
        z_var = self.relu2(self.batch_norm2(self.convtran2(z_var)))

        self.convtran3 = tf.keras.layers.Conv2DTranspose(
            filters=self.g_dim // 2, kernel_size=3, strides=(2, 2), padding="SAME")
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()
        z_var = self.relu3(self.batch_norm3(self.convtran3(z_var)))

        self.convtran4 = tf.keras.layers.Conv2DTranspose(
            filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation='sigmoid')
        z_var = self.convtran4(z_var)
        return z_var

    def build_hr_generator_woc(self):
        '''
        without conditions
        input: stageI generate sample (N, 128, 128, 1)
        '''
        in_var = tf.keras.Input(shape=self.generate_shape_I)
        x_var = self.hr_g_encode_sample(in_var)  # s4 * s4 * g_dim * 4 : (N, 32, 32, 128)

        # Joint learning from text and image -> s4 * s4 * gf_dim * 4
        node0 = self.hr_g_joint(x_var)
        node1 = self.residual_block(node0)
        node2 = self.residual_block(node1)
        node3 = self.residual_block(node2)
        node4 = self.residual_block(node3)

        # Up-sampling
        generate_var = self.hr_generator_simple(node4)
        self.hr_generator_wo_condition = tf.keras.models.Model(inputs=in_var, outputs=generate_var)

    def hr_generate_woc(self, in_var):
        '''
        without condition
        '''
        generate_var = self.hr_generator_wo_condition(in_var)
        return generate_var

    def build_hr_generator_wca(self):
        x_code = tf.keras.Input(shape=self.generate_shape_I)
        c_code = tf.keras.Input(shape=(self.c_dim))  # (N, 1042)
        cin_var = [c_code, x_code]

        x_var = self.hr_g_encode_sample(x_code)  # (N, 32, 32, 128)

        tuple_x = self.condition(c_code)
        c_eps = self.reparameterize(tuple_x)  # (N, 100)
        c_var = tf.expand_dims(tf.expand_dims(c_eps, 1), 1)  # shape=(N, 1, 1, 100)
        c_var = tf.tile(c_var, [1, self.s4, self.s4, 1])  # shape=(N, 32, 32, 100)

        # combine both -> s4 * s4 * (ef_dim + gf_dim * 4)
        cx_var = tf.concat([c_var, x_var], 3)  # shape=(N, 32, 32, 228)

        # Joint learning from text and image -> s4 * s4 * gf_dim * 4
        node0 = self.hr_g_joint(cx_var)
        node1 = self.residual_block(node0)
        node2 = self.residual_block(node1)
        node3 = self.residual_block(node2)
        node4 = self.residual_block(node3)

        generate_var = self.hr_generator_simple(node4)
        self.hr_generator_wi_conaugment = tf.keras.models.Model(inputs=cin_var, outputs=[generate_var, tuple_x])

    def hr_generate_wca(self, c_var, x_var):
        generate_var, tuple_x = self.hr_generator_wi_conaugment([c_var, x_var])
        return generate_var, tuple_x


class Discriminator(tf.keras.Model):
    def __init__(self, c_dim, d_dim, ef_dim, generate_shape_I, generate_shape_II):
        super(Discriminator, self).__init__()
        self.c_dim = c_dim
        self.d_dim = d_dim
        self.ef_dim = ef_dim
        self.generate_shape_I = generate_shape_I
        self.generate_shape_II = generate_shape_II
        
        self.s = self.generate_shape_I[0]
        self.s2, self.s4, self.s8, self.s16 = int(
            self.s / 2), int(self.s / 4), int(self.s / 8), int(self.s / 16)


    def context_embedding(self, c_var):
        '''
        In: (N, 1024)
        Out: (N, 100)
        '''
        self.fc = tf.keras.layers.Dense(self.ef_dim)
        c_var = self.fc(c_var)
        self.ac = tf.keras.layers.LeakyReLU(alpha=0.2)
        c_var = self.ac(c_var)
        return c_var

    def d_encode_sample(self, x_var):
        '''
        in: 128 * 128 * 1
        out: 8 * 8 * 64
        '''
        self.conv1 = tf.keras.layers.Conv2D(
            filters=self.d_dim, kernel_size=3, strides=(2, 2), padding="SAME")
        self.ac1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        x_var = self.ac1(self.conv1(x_var))

        self.conv2 = tf.keras.layers.Conv2D(
            filters=self.d_dim * 2, kernel_size=3, strides=(2, 2), padding="SAME")
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.ac2 = tf.keras.layers.LeakyReLU(alpha=0.2)
        x_var = self.ac2(self.batch_norm2(self.conv2(x_var)))

        self.conv3 = tf.keras.layers.Conv2D(
            filters=self.d_dim * 4, kernel_size=3, strides=(2, 2), padding="SAME")
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.ac3 = tf.keras.layers.LeakyReLU(alpha=0.2)
        x_var = self.ac3(self.batch_norm3(self.conv3(x_var)))

        self.conv4 = tf.keras.layers.Conv2D(
            filters=self.d_dim * 8, kernel_size=3, strides=(2, 2), padding="SAME")
        self.batch_norm4 = tf.keras.layers.BatchNormalization()
        self.ac4 = tf.keras.layers.LeakyReLU(alpha=0.2)
        x_var = self.ac4(self.batch_norm4(self.conv4(x_var)))

        return x_var

    def hr_d_encode_sample(self, x_var):
        '''
        in: 256 * 256 * 1 <--> 2s * 2s * 1
        out: 8 * 8 * 64
        '''
        self.conv1 = tf.keras.layers.Conv2D(
            filters=self.d_dim, kernel_size=3, strides=(2, 2), padding="SAME")
        self.ac1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        x_var = self.ac1(self.conv1(x_var))

        self.conv2 = tf.keras.layers.Conv2D(
            filters=self.d_dim * 2, kernel_size=3, strides=(2, 2), padding="SAME")
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.ac2 = tf.keras.layers.LeakyReLU(alpha=0.2)
        x_var = self.ac2(self.batch_norm2(self.conv2(x_var)))

        self.conv3 = tf.keras.layers.Conv2D(
            filters=self.d_dim * 4, kernel_size=3, strides=(2, 2), padding="SAME")
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.ac3 = tf.keras.layers.LeakyReLU(alpha=0.2)
        x_var = self.ac3(self.batch_norm3(self.conv3(x_var)))

        self.conv4 = tf.keras.layers.Conv2D(
            filters=self.d_dim * 8, kernel_size=3, strides=(2, 2), padding="SAME")
        self.batch_norm4 = tf.keras.layers.BatchNormalization()
        self.ac4 = tf.keras.layers.LeakyReLU(alpha=0.2)
        x_var = self.ac4(self.batch_norm4(self.conv4(x_var)))

        self.conv5 = tf.keras.layers.Conv2D(
            filters=self.d_dim * 16, kernel_size=3, strides=(2, 2), padding="SAME")
        self.batch_norm5 = tf.keras.layers.BatchNormalization()
        self.ac5 = tf.keras.layers.LeakyReLU(alpha=0.2)
        x_var = self.ac5(self.batch_norm5(self.conv5(x_var)))

        self.conv6 = tf.keras.layers.Conv2D(
            filters=self.d_dim * 8, kernel_size=1, strides=(1, 1), padding="SAME")
        self.batch_norm6 = tf.keras.layers.BatchNormalization()
        x_var_0 = self.batch_norm6(self.conv6(x_var))

        # 残差块
        self.conv7 = tf.keras.layers.Conv2D(
            filters=self.d_dim * 8, kernel_size=1, strides=(1, 1), padding="SAME")
        self.batch_norm7 = tf.keras.layers.BatchNormalization()
        self.ac7 = tf.keras.layers.LeakyReLU(alpha=0.2)
        x_var = self.ac7(self.batch_norm7(self.conv7(x_var_0)))

        self.conv8 = tf.keras.layers.Conv2D(
            filters=self.d_dim * 8, kernel_size=3, strides=(1, 1), padding="SAME")
        self.batch_norm8 = tf.keras.layers.BatchNormalization()
        x_var_1 = self.batch_norm8(self.conv8(x_var))

        x_var_10 = tf.add(x_var_0, x_var_1)
        self.ac9 = tf.keras.layers.LeakyReLU(alpha=0.2)
        x_var_10 = self.ac9(x_var_10)
        return x_var_10

    def discriminator_simple(self, var, out_shape):
        '''
        in: (N, 8, 8, 164) / (N, 8, 8, 64) -> 8 * 8 * 128
        out: out_shape = 1 or 4
        '''
        self.conv1 = tf.keras.layers.Conv2D(
            filters=self.d_dim * 16, kernel_size=1, strides=(1, 1), padding="SAME")
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.ac1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        var = self.ac1(self.batch_norm1(self.conv1(var)))

        self.conv2 = tf.keras.layers.Conv2D(
            filters=self.d_dim * 8, kernel_size=3, strides=(2, 2), padding="SAME")
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.ac2 = tf.keras.layers.LeakyReLU(alpha=0.2)
        var = self.ac2(self.batch_norm2(self.conv2(var)))

        self.conv3 = tf.keras.layers.Conv2D(
            filters=self.d_dim * 4, kernel_size=3, strides=(2, 2), padding="SAME")
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.ac3 = tf.keras.layers.LeakyReLU(alpha=0.2)
        var = self.ac3(self.batch_norm3(self.conv3(var)))

        self.flatten = tf.keras.layers.Flatten()
        var = self.flatten(var)

        if out_shape == 1:
            self.fc = tf.keras.layers.Dense(out_shape, activation='sigmoid')
        else:
            self.fc = tf.keras.layers.Dense(out_shape, activation='softmax')
        var = self.fc(var)
        return var

    def build_discriminator_woc(self):
        '''
        without condition
        '''
        x_var = tf.keras.Input(shape=self.generate_shape_I)
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
        x_var = tf.keras.Input(shape=self.generate_shape_I)
        c_var = tf.keras.Input(shape=self.c_dim)
        in_var = [c_var, x_var]

        x_code = self.d_encode_sample(x_var)
        c_code = self.context_embedding(c_var)
        c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1)  # shape=(N, 1, 1, 100)
        c_code = tf.tile(c_code, [1, self.s16, self.s16, 1])  # shape=(N, 8, 8, 100)

        x_c_code = tf.concat([x_code, c_code], 3)  # shape=(N, 8, 8, 164)

        var = self.discriminator_simple(x_c_code, 3)
        # 生成的sample为假  生成的sample为真与条件不匹配  生成的sample为真与条件匹配
        self.discriminator_wi_condition = tf.keras.models.Model(inputs=in_var, outputs=var)

    def discriminate_wc(self, in_var):
        '''
        with condition
        '''
        discriminate_var = self.discriminator_wi_condition(in_var)
        return discriminate_var

    def build_hr_discriminator_woc(self):
        x_var = tf.keras.Input(shape=self.generate_shape_II)
        x_code = self.hr_d_encode_sample(x_var)
        var = self.discriminator_simple(x_code, 1)
        self.hr_discriminator_wo_condition = tf.keras.models.Model(inputs=x_var, outputs=var)

    def hr_discriminate_woc(self, x_var):
        discriminate_var = self.hr_discriminator_wo_condition(x_var)
        return discriminate_var

    def build_hr_discriminator_wc(self):
        x_var = tf.keras.Input(shape=self.generate_shape_II)
        c_var = tf.keras.Input(shape=self.c_dim)
        in_var = [c_var, x_var]

        x_code = self.hr_d_encode_sample(x_var)  # (N, 8, 8, 64)
        c_code = self.context_embedding(c_var)  # (N, 100)
        c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1)
        c_code = tf.tile(c_code, [1, self.s16, self.s16, 1])  # (N, 8, 8, 100)

        x_c_code = tf.concat([x_code, c_code], 3)  # (N, 8, 8, 164)

        var = self.discriminator_simple(x_c_code, 3)
        self.hr_discriminator_wi_condition = tf.keras.models.Model(inputs=in_var, outputs=var)

    def hr_discriminate_wc(self, in_var):
        discriminate_var = self.hr_discriminator_wi_condition(in_var)
        return discriminate_var


class StageII(tf.keras.Model):
    def __init__(self, batch_size, d_dim, g_dim, z_dim, c_dim, ef_dim, generate_shape_I):
        super(StageII, self).__init__()

        self.batch_size = batch_size
        self.d_dim = d_dim
        self.g_dim = g_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.ef_dim = ef_dim
        self.generate_shape_I = generate_shape_I

        self.generator_II = Generator(self.batch_size, self.g_dim, self.z_dim, self.c_dim, self.ef_dim, self.generate_shape_I, self.generate_shape_II)
        self.discriminator_II = Discriminator(self.d_dim, self.ef_dim, self.generate_shape_I, self.generate_shape_II)