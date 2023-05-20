import os
import tensorflow as tf
from datetime import datetime
from nn.trainer import Trainer


class WassersteinTrainer(Trainer):
    def train_step_g_woc(self):
        log_vars = []
        with tf.GradientTape() as tape:
            fake_lr_seeds = self.scale_model.model.generate_woc()
            fake_logit = self.scale_model.model.discriminate_woc(fake_lr_seeds)
            g_loss_fake = - tf.reduce_mean(fake_logit)

        log_vars.append(("g_loss_fake", g_loss_fake))
        gradients = tape.gradient(g_loss_fake, self.scale_model.generator.trainable_variables)
        self.optimizer.stageI_g_opt.apply_gradients(zip(gradients, self.scale_model.generator.trainable_variables))
        return log_vars

    def train_step_g_wc(self, c):
        log_vars = []
        with tf.GradientTape() as tape:
            fake_lr_seeds = self.scale_model.model.generate_wc(c)
            fake_logit = self.scale_model.model.discriminate_wc([c, fake_lr_seeds])
            g_loss_fake = - tf.reduce_mean(fake_logit)
        
        log_vars.append(("g_loss_fake", g_loss_fake))
        gradients = tape.gradient(g_loss_fake, self.scale_model.generator.trainable_variables)
        self.optimizer.stageI_g_opt.apply_gradients(zip(gradients, self.scale_model.generator.trainable_variables))
        return log_vars

    def train_step_g_wca(self, c):
        '''
        args:
            c: 条件edges
        '''
        log_vars = []
        with tf.GradientTape() as tape:
            fake_lr_seeds, tuple_x = self.scale_model.model.generate_wca(c)
            fake_logit = self.scale_model.model.discriminate_wc([c, fake_lr_seeds])
            g_loss_fake = - tf.reduce_mean(fake_logit)

            mean, log_sigma = tuple_x
            kl_loss = self.count_klloss(mean, log_sigma)
            generator_loss = g_loss_fake + self.lambda_kl * kl_loss
        
        log_vars.append(("g_loss_fake", g_loss_fake))
        log_vars.append(("kl_loss", kl_loss))
        log_vars.append(("generator_loss", generator_loss))

        gradients = tape.gradient(generator_loss, self.scale_model.generator.trainable_variables)
        self.optimizer.stageI_g_opt.apply_gradients(zip(gradients, self.scale_model.generator.trainable_variables))
        return log_vars

    def train_step_d_woc(self, lr_seeds):
        log_vars = []
        with tf.GradientTape() as tape:
            fake_lr_seeds = self.scale_model.model.generate_woc()
            real_logit = self.scale_model.model.discriminate_woc(lr_seeds)
            fake_logit = self.scale_model.model.discriminate_woc(fake_lr_seeds)
            
            # alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
            alpha = tf.random.uniform(lr_seeds.shape, 0.0, 1.0)
            inter_seeds = fake_lr_seeds * alpha + lr_seeds * (1 - alpha)
            with tf.GradientTape() as tape_gp:
                tape_gp.watch(inter_seeds)
                inter_logit = self.scale_model.model.discriminate_woc(inter_seeds)
            gp_gradients = tape_gp.gradient(inter_logit, inter_seeds)
            # print("gp_gradients.shape:", gp_gradients.shape)

            gp_gradients_norm = tf.sqrt(tf.reduce_sum(
                tf.square(gp_gradients), axis=[1]))
            
            # print("gp_gradients_norm.shape:", gp_gradients_norm.shape)
            # print("gp_gradients_norm:", gp_gradients_norm)
            # return 0
            gp = tf.reduce_mean((gp_gradients_norm - 1.0) ** 2)

            d_loss_real = - tf.reduce_mean(real_logit)
            d_loss_fake = tf.reduce_mean(fake_logit)

            discriminator_loss = d_loss_real + d_loss_fake + gp * self.lambda_gp

            log_vars.append(("d_loss_real", d_loss_real))
            log_vars.append(("d_loss_fake", d_loss_fake))
            log_vars.append(("discriminator_loss", discriminator_loss))

        gradients = tape.gradient(discriminator_loss, self.scale_model.discriminator.trainable_variables)
        self.optimizer.stageI_d_opt.apply_gradients(zip(gradients, self.scale_model.discriminator.trainable_variables))
        return log_vars

    def train_step_d_wc(self, lr_seeds, c, w_c):
        log_vars = []
        with tf.GradientTape() as tape:
            real_logit = self.scale_model.model.discriminate_wc([c, lr_seeds])
            fake_lr_seeds = self.scale_model.model.generate_wc(c)
            fake_logit = self.scale_model.model.discriminate_wc([c, fake_lr_seeds])
            wrong_logit = self.scale_model.model.discriminate_wc([w_c, lr_seeds])
            
            # alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
            alpha = tf.random.uniform(lr_seeds.shape, 0.0, 1.0)
            inter_seeds = fake_lr_seeds * alpha + lr_seeds * (1 - alpha)
            with tf.GradientTape() as tape_gp:
                tape_gp.watch(inter_seeds)
                inter_logit = self.scale_model.model.discriminate_wc([c, inter_seeds])
            gp_gradients = tape_gp.gradient(inter_logit, inter_seeds)
            gp_gradients_norm = tf.sqrt(tf.reduce_sum(
                tf.square(gp_gradients), axis=[1]))
            gp = tf.reduce_mean((gp_gradients_norm - 1.0) ** 2)

            d_loss_real = - tf.reduce_mean(real_logit)
            d_loss_fake = tf.reduce_mean(fake_logit)
            d_loss_wrong = tf.reduce_mean(wrong_logit)

            discriminator_loss = 0.5 * (d_loss_fake + d_loss_wrong) + d_loss_real + gp * self.lambda_gp

            log_vars.append(("d_loss_real", d_loss_real))
            log_vars.append(("d_loss_fake", d_loss_fake))
            log_vars.append(("d_loss_wrong", d_loss_wrong))
            log_vars.append(("discriminator_loss", discriminator_loss))

        gradients = tape.gradient(discriminator_loss, self.scale_model.discriminator.trainable_variables)
        self.optimizer.stageI_d_opt.apply_gradients(zip(gradients, self.scale_model.discriminator.trainable_variables))
        return log_vars

    def train_step_d_wca(self, lr_seeds, c, w_c):
        log_vars = []
        with tf.GradientTape() as tape:
            real_logit = self.scale_model.model.discriminate_wc([c, lr_seeds])
            fake_lr_seeds, tuple_x = self.scale_model.model.generate_wca(c)
            fake_logit = self.scale_model.model.discriminate_wc([c, fake_lr_seeds])
            wrong_logit = self.scale_model.model.discriminate_wc([w_c, lr_seeds])
            
            # alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
            alpha = tf.random.uniform(lr_seeds.shape, 0.0, 1.0)
            inter_seeds = fake_lr_seeds * alpha + lr_seeds * (1 - alpha)
            with tf.GradientTape() as tape_gp:
                tape_gp.watch(inter_seeds)
                inter_logit = self.scale_model.model.discriminate_wc([c, inter_seeds])
            gp_gradients = tape_gp.gradient(inter_logit, inter_seeds)
            gp_gradients_norm = tf.sqrt(tf.reduce_sum(
                tf.square(gp_gradients), axis=[1]))
            gp = tf.reduce_mean((gp_gradients_norm - 1.0) ** 2)

            d_loss_real = - tf.reduce_mean(real_logit)
            d_loss_fake = tf.reduce_mean(fake_logit)
            d_loss_wrong = tf.reduce_mean(wrong_logit)

            discriminator_loss = 0.5 * (d_loss_fake + d_loss_wrong) + d_loss_real + gp * self.lambda_gp

            log_vars.append(("d_loss_real", d_loss_real))
            log_vars.append(("d_loss_fake", d_loss_fake))
            log_vars.append(("d_loss_wrong", d_loss_wrong))
            log_vars.append(("discriminator_loss", discriminator_loss))

        gradients = tape.gradient(discriminator_loss, self.scale_model.discriminator.trainable_variables)
        self.optimizer.stageI_d_opt.apply_gradients(zip(gradients, self.scale_model.discriminator.trainable_variables))
        return log_vars