import os
import sys
import tensorflow as tf
from datetime import datetime
from nn.trainer import Trainer


class CrossEntropyTrainer(Trainer):
    def __init__(self, train_data, model, flag, train_ratio_I, train_ratio_II, max_epoch, optimizer, decay_epochs, generate_shape_I, generate_shape_II, log_dir, model_path, lambda_0=1):
        self.train_data = train_data
        self.model = model
        self.flag = flag
        self.max_epoch = max_epoch

        self.model_path = model_path
        self.log_dir = log_dir
        self.train_ratio_I = train_ratio_I
        self.train_ratio_II = train_ratio_II
        self.lambda_0 = lambda_0

        self.decay_epochs = decay_epochs
        
        self.optimizer = optimizer

        self.generate_shape_I = generate_shape_I
        self.generate_shape_II = generate_shape_II
        
        now = datetime.now()
        self.date_time = now.strftime("%m%d%H%M%S")

        self.log_name = 'train_' + self.date_time
        self.save_log = os.path.join(self.log_dir, self.log_name)
        self.log = open(self.save_log, "a")

        self.batch_size = self.model.batch_size

        real_label = tf.Variable([1, 0, 0], dtype=tf.float32)
        real_label = tf.expand_dims(real_label, 0)
        self.real_label = tf.tile(real_label, [self.batch_size, 1])

        wrong_label = tf.Variable([0, 1, 0], dtype=tf.float32)
        wrong_label = tf.expand_dims(wrong_label, 0)
        self.wrong_label = tf.tile(wrong_label, [self.batch_size, 1])

        fake_label = tf.Variable([0, 0, 1], dtype=tf.float32)
        fake_label = tf.expand_dims(fake_label, 0)
        self.fake_label = tf.tile(fake_label, [self.batch_size, 1])
    
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

    def train_step_woc(self, lr_seeds, i):
        '''
        In: model, lr_seeds
        Out: discriminator_loss, generator_loss
        Loss:
        cross entropy:
        fake_lr_seeds = self.model.generator_I.generate_woc()
        real_logit = self.model.discriminator_I.discriminate_woc(lr_seeds)
        fake_logit = self.model.discriminator_I.discriminate_woc(fake_lr_seeds)
        d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit, labels=tf.ones_like(real_logit))
        d_loss_real = tf.reduce_mean(d_loss_real)
        d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.zeros_like(fake_logit))
        d_loss_fake = tf.reduce_mean(d_loss_fake)
        discriminator_loss = d_loss_real + d_loss_fake
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.ones_like(fake_logit))
        generator_loss = tf.reduce_mean(generator_loss)

        g_loss_fake: 0.6904571056365967(0.69) d_loss_real: 0.33304929733276367(0) d_loss_fake: 0.6958579421043396(0.69)

        log_vars.append(("g_loss_fake", generator_loss))
        log_vars.append(("d_loss_real", d_loss_real))
        log_vars.append(("d_loss_fake", d_loss_fake))

        without cross entropy:
        fake_lr_seeds = self.model.generator_I.generate_woc()
        real_pred = self.model.discriminator_I.discriminate_woc(lr_seeds)
        fake_pred = self.model.discriminator_I.discriminate_woc(fake_lr_seeds)
        d_loss_real = - tf.reduce_mean(real_pred)
        d_loss_fake = tf.reduce_mean(fake_pred)
        discriminator_loss = d_loss_real + d_loss_fake
        generator_loss = - tf.reduce_mean(fake_pred)
        
        g_loss_fake: -0.5578884(-0.5) d_loss_real: -0.94022703(-1) d_loss_fake: 0.5578884(0.5)
        '''
        log_vars = []
        with tf.GradientTape() as lr_gen_tape, tf.GradientTape() as lr_disc_tape:

            fake_lr_seeds = self.model.generator_I.generate_woc()
            real_logit = self.model.discriminator_I.discriminate_woc(lr_seeds)
            fake_logit = self.model.discriminator_I.discriminate_woc(fake_lr_seeds)

            d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit, labels=tf.ones_like(real_logit))
            d_loss_real = tf.reduce_mean(d_loss_real)
            d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.zeros_like(fake_logit))
            d_loss_fake = tf.reduce_mean(d_loss_fake)
            discriminator_loss = d_loss_real + d_loss_fake
            generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.ones_like(fake_logit))
            generator_loss = tf.reduce_mean(generator_loss)

            log_vars.append(("g_loss_fake", generator_loss))
            log_vars.append(("d_loss_real", d_loss_real))
            log_vars.append(("d_loss_fake", d_loss_fake))

        if i % self.train_ratio_I != 0:
            generator_gradients = lr_gen_tape.gradient(generator_loss, self.model.generator_I.trainable_variables)
            self.optimizer.stageI_g_opt.apply_gradients(zip(generator_gradients, self.model.generator_I.trainable_variables))
        else:
            discriminator_gradients = lr_disc_tape.gradient(discriminator_loss, self.model.discriminator_I.trainable_variables)
            self.optimizer.stageI_d_opt.apply_gradients(zip(discriminator_gradients, self.model.discriminator_I.trainable_variables))

        return log_vars

    def hr_train_step_woc(self, hr_seeds, i):
        '''
        In: model, hr_seeds
        Out: discriminator_loss, generator_loss
        '''
        log_vars = []
        with tf.GradientTape() as hr_genI_tape, tf.GradientTape() as hr_genII_tape, tf.GradientTape() as hr_disc_tape:
            fake_lr_seeds = self.model.generator_I.generate_woc()
            fake_hr_seeds = self.model.generator_II.hr_generate_woc(fake_lr_seeds)

            real_logit = self.model.discriminator_II.hr_discriminate_woc(hr_seeds)
            fake_logit = self.model.discriminator_II.hr_discriminate_woc(fake_hr_seeds)

            d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit, labels=tf.ones_like(real_logit))
            d_loss_real = tf.reduce_mean(d_loss_real)
            d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.zeros_like(fake_logit))
            d_loss_fake = tf.reduce_mean(d_loss_fake)
            discriminator_loss = d_loss_real + d_loss_fake
            generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.ones_like(fake_logit))
            generator_loss = tf.reduce_mean(generator_loss)

            log_vars.append(("hr_g_loss_fake", generator_loss))
            log_vars.append(("hr_d_loss_real", d_loss_real))
            log_vars.append(("hr_d_loss_fake", d_loss_fake))

        if i % self.train_ratio_II != 0:
            generatorI_gradients = hr_genI_tape.gradient(generator_loss, self.model.generator_I.trainable_variables)
            self.optimizer.stageII_gI_opt.apply_gradients(zip(generatorI_gradients, self.model.generator_I.trainable_variables))
            generatorII_gradients = hr_genII_tape.gradient(generator_loss, self.model.generator_II.trainable_variables)
            self.optimizer.stageII_gII_opt.apply_gradients(zip(generatorII_gradients, self.model.generator_II.trainable_variables))
        else:
            discriminator_gradients = hr_disc_tape.gradient(discriminator_loss, self.model.discriminator_II.trainable_variables)
            self.optimizer.stageII_dII_opt.apply_gradients(zip(discriminator_gradients, self.model.discriminator_II.trainable_variables))

        return log_vars

    def train_step_wc(self, lr_seeds):
        # compute_loss_wc(lr_seeds, wrong_lr_seeds, fake_lr_seeds, embeddings)
        pass

    def hr_train_step_wc(self, hr_seeds):
        pass

    def train_step_wca(self, lr_seeds, condition, wrong_condition, i):
        log_vars = []
        with tf.GradientTape() as lr_gen_tape, tf.GradientTape() as lr_disc_tape:

            fake_lr_seeds, tuple_x = self.model.generator_I.generate_wca(condition)
            real_pred = self.model.discriminator_I.discriminate_wc([condition, lr_seeds])
            fake_pred = self.model.discriminator_I.discriminate_wc([condition, fake_lr_seeds])
            wrong_pred = self.model.discriminator_I.discriminate_wc([wrong_condition, lr_seeds])

            d_loss_real = self.cross_entropy(self.real_label, real_pred, 0)
            d_loss_wrong = self.cross_entropy(self.wrong_label, wrong_pred, 1)
            d_loss_fake = self.cross_entropy(self.fake_label, fake_pred, 2)
            discriminator_loss = d_loss_real + (d_loss_fake + d_loss_wrong) / 2

            mean, log_sigma = tuple_x
            kl_loss = self.count_klloss(mean, log_sigma)
            generator_loss = self.cross_entropy(self.real_label, fake_pred, 0) + self.lambda_0 * kl_loss
        
            log_vars.append(("kl_loss", kl_loss))
            log_vars.append(("g_loss", generator_loss - self.lambda_0 * kl_loss))
            log_vars.append(("generator_loss", generator_loss))
            log_vars.append(("d_loss_real", d_loss_real))
            log_vars.append(("d_loss_fake", d_loss_fake))
            log_vars.append(("d_loss_wrong", d_loss_wrong))
            log_vars.append(("discriminator_loss", discriminator_loss))

        if i % self.train_ratio_I != 0:
            generator_gradients = lr_gen_tape.gradient(generator_loss, self.model.generator_I.trainable_variables)
            self.optimizer.stageI_g_opt.apply_gradients(zip(generator_gradients, self.model.generator_I.trainable_variables))
        else:
            discriminator_gradients = lr_disc_tape.gradient(discriminator_loss, self.model.discriminator_I.trainable_variables)
            self.optimizer.stageI_d_opt.apply_gradients(zip(discriminator_gradients, self.model.discriminator_I.trainable_variables))

        return log_vars

    def hr_train_step_wca(self, hr_seeds, condition, wrong_condition, i):
        log_vars = []
        with tf.GradientTape() as hr_genI_tape, tf.GradientTape() as hr_genII_tape, tf.GradientTape() as hr_disc_tape:

            fake_lr_seeds, tuple_x = self.model.generator_I.generate_wca(condition)
            fake_hr_seeds, hr_tuple_x = self.model.generator_II.hr_generate_wca(condition, fake_lr_seeds)

            real_pred = self.model.discriminator_II.hr_discriminate_wc([condition, hr_seeds])
            wrong_pred = self.model.discriminator_II.hr_discriminate_wc([wrong_condition, hr_seeds])
            fake_pred = self.model.discriminator_II.hr_discriminate_wc([condition, fake_hr_seeds])

            d_loss_real = self.cross_entropy(self.real_label, real_pred, 0)
            d_loss_wrong = self.cross_entropy(self.wrong_label, wrong_pred, 1)
            d_loss_fake = self.cross_entropy(self.fake_label, fake_pred, 2)
            discriminator_loss = d_loss_real + (d_loss_fake + d_loss_wrong) / 2

            mean, log_sigma = hr_tuple_x
            kl_loss = self.count_klloss(mean, log_sigma)
            generator_loss = self.cross_entropy(self.real_label, fake_pred, 0) + self.lambda_0 * kl_loss
        
            log_vars.append(("hr_kl_loss", kl_loss))
            log_vars.append(("hr_g_loss", generator_loss - kl_loss))
            log_vars.append(("hr_generator_loss", generator_loss))
            log_vars.append(("hr_d_loss_real", d_loss_real))
            log_vars.append(("hr_d_loss_fake", d_loss_fake))
            log_vars.append(("hr_d_loss_wrong", d_loss_wrong))
            log_vars.append(("hr_discriminator_loss", discriminator_loss))

        if i % self.train_ratio_II != 0:
            generatorI_gradients = hr_genI_tape.gradient(generator_loss, self.model.generator_I.trainable_variables)
            self.optimizer.stageII_gI_opt.apply_gradients(zip(generatorI_gradients, self.model.generator_I.trainable_variables))
            generatorII_gradients = hr_genII_tape.gradient(generator_loss, self.model.generator_II.trainable_variables)
            self.optimizer.stageII_gII_opt.apply_gradients(zip(generatorII_gradients, self.model.generator_II.trainable_variables))
        else:
            discriminator_gradients = hr_disc_tape.gradient(discriminator_loss, self.model.discriminator_II.trainable_variables)
            self.optimizer.stageII_dII_opt.apply_gradients(zip(discriminator_gradients, self.model.discriminator_II.trainable_variables))

        return log_vars

    def train_I(self):
        i = 0
        for epoch in range(self.max_epoch):
            for lr_seeds, hr_seeds, condition, wrong_condition in self.train_data:
                reshape_I = tf.keras.layers.Reshape(self.generate_shape_I)
                lr_seeds = reshape_I(lr_seeds)

                if self.flag == 'LWOC':
                    log_vars = self.train_step_woc(lr_seeds, i)
                elif self.flag == 'LWCA':
                    log_vars = self.train_step_wca(lr_seeds, condition, wrong_condition, i)
                else:
                    pass
                i += 1
            
            self.log.write('Epoch: {}'.format(epoch + 1))
            for k, v in log_vars:
                self.log.write(' {}: {}'.format(k, v))
            self.log.write('\n')
            self.log.flush()

            # 学习率更新
            if (epoch + 1) % self.decay_epochs == 0:
                self.optimizer.lr_decay()

    def train_II(self):
        i = 0
        for epoch in range(self.max_epoch, 2 * self.max_epoch):
            for lr_seeds, hr_seeds, condition, wrong_condition in self.train_data:
                reshape_II = tf.keras.layers.Reshape(self.generate_shape_II)
                hr_seeds = reshape_II(hr_seeds)

                if self.flag == 'LWOC':
                    log_vars = self.hr_train_step_woc(hr_seeds, i)
                elif self.flag == 'LWCA':
                    log_vars = self.hr_train_step_wca(hr_seeds, condition, wrong_condition, i)
                else:
                    pass
                i += 1

            self.log.write('Epoch: {}'.format(epoch + 1))
            for k, v in log_vars:
                self.log.write(' {}: {}'.format(k, v))
            self.log.write('\n')
            self.log.flush()
            # 学习率更新
            if (epoch + 1) % self.decay_epochs == 0:
                self.optimizer.lr_decay('II')

            # self.log.write('stageII_gI_lr: {} stageII_gII_lr: {} stageII_dII_lr: {}\n'.format(self.optimizer.stageII_gI_opt.learning_rate, self.optimizer.stageII_gII_opt.learning_rate, self.optimizer.stageII_dII_opt.learning_rate))
            # self.log.flush()

    def train(self):
        i = 0
        for epoch in range(self.max_epoch):
            for lr_seeds, hr_seeds, condition, wrong_condition in self.train_data:
                # condition wrong_condition
                reshape_I = tf.keras.layers.Reshape(self.generate_shape_I)
                lr_seeds = reshape_I(lr_seeds)

                if self.flag == 'LWOC':
                    log_vars = self.train_step_woc(lr_seeds, i)
                elif self.flag == 'LWCA':
                    log_vars = self.train_step_wca(lr_seeds, condition, wrong_condition, i)
                else:
                    pass
                i += 1
            
            self.log.write('Epoch: {}'.format(epoch + 1))
            for k, v in log_vars:
                self.log.write(' {}: {}'.format(k, v))
            self.log.write('\n')
            self.log.flush()
            # 学习率更新
            if (epoch + 1) % self.decay_epochs == 0:
                self.optimizer.lr_decay()

        for epoch in range(self.max_epoch, 2 * self.max_epoch):
            for lr_seeds, hr_seeds, condition, wrong_condition in self.train_data:
                # condition wrong_condition
                reshape_II = tf.keras.layers.Reshape(self.generate_shape_II)
                hr_seeds = reshape_II(hr_seeds)

                if self.flag == 'LWOC':
                    log_vars = self.hr_train_step_woc(hr_seeds, i)
                elif self.flag == 'LWCA':
                    log_vars = self.hr_train_step_wca(hr_seeds, condition, wrong_condition, i)
                else:
                    pass
                i += 1

            self.log.write('Epoch: {}'.format(epoch + 1))
            for k, v in log_vars:
                self.log.write(' {}: {}'.format(k, v))
            self.log.write('\n')
            self.log.flush()
            # 学习率更新
            if (epoch + 1) % self.decay_epochs == 0:
                self.optimizer.lr_decay('II')

    def save_model_I(self):
        submodel_names = ['generator_I', 'discriminator_I']
        submodels = [self.model.generator_I, self.model.discriminator_I]
        for i in range(2):
            submodel_name = submodel_names[i]
            submodel = submodels[i]
            # datetime + wocondition/wirefuse/wiedges + generator/discriminator
            model_name = '{}_{}_{}_{}'.format(self.date_time, self.model_type, self.flag, submodel_name)
            model_path = self.model_path + '/{}'.format(model_name)
            submodel.save_weights(model_path)

    def save_model_II(self):
        submodel_names = ['generator_II', 'discriminator_II']
        submodels = [self.model.generator_II, self.model.discriminator_II]
        for i in range(2):
            submodel_name = submodel_names[i]
            submodel = submodels[i]
            # datetime + wocondition/wirefuse/wiedges + generator/discriminator
            model_name = '{}_{}_{}_{}'.format(self.date_time, self.model_type, self.flag, submodel_name)
            model_path = self.model_path + '/{}'.format(model_name)
            submodel.save_weights(model_path)

    def save_model(self):
        submodel_names = ['generator_I', 'discriminator_I', 'generator_II', 'discriminator_II']
        submodels = [self.model.generator_I, self.model.discriminator_I, self.model.generator_II, self.model.discriminator_II]
        for i in range(4):
            submodel_name = submodel_names[i]
            submodel = submodels[i]
            # datetime + wocondition/wirefuse/wiedges + generator/discriminator
            model_name = '{}_{}_{}_{}'.format(self.date_time, self.model_type, self.flag, submodel_name)
            model_path = self.model_path + '/{}'.format(model_name)
            submodel.save_weights(model_path)




