import tensorflow as tf


# reduce_mean normalize also the dimension of the embeddings
def kl_loss(mu, log_sigma):
    loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
    loss = tf.reduce_mean(loss)
    return loss


def compute_losses(self, images, wrong_images, fake_images, embeddings):
    '''
    In: real_images, wrong_images, fake_images, embeddings
    Out: discriminator_loss, generator_loss
    '''
    real_logit = self.model.get_discriminator(images, embeddings)
    wrong_logit = self.model.get_discriminator(wrong_images, embeddings)
    fake_logit = self.model.get_discriminator(fake_images, embeddings)

    real_d_loss = tf.nn.sigmoid_cross_entropy_with_logits(real_logit, tf.ones_like(real_logit))
    real_d_loss = tf.reduce_mean(real_d_loss)
    wrong_d_loss = tf.nn.sigmoid_cross_entropy_with_logits(wrong_logit, tf.zeros_like(wrong_logit))
    wrong_d_loss = tf.reduce_mean(wrong_d_loss)
    fake_d_loss = tf.nn.sigmoid_cross_entropy_with_logits(fake_logit, tf.zeros_like(fake_logit))
    fake_d_loss = tf.reduce_mean(fake_d_loss)
    
    if cfg.TRAIN.B_WRONG:
        discriminator_loss = real_d_loss + (wrong_d_loss + fake_d_loss) / 2.
        self.log_vars.append(("d_loss_wrong", wrong_d_loss))
    else:
        discriminator_loss = real_d_loss + fake_d_loss
    self.log_vars.append(("d_loss_real", real_d_loss))
    self.log_vars.append(("d_loss_fake", fake_d_loss))

    generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(fake_logit, tf.ones_like(fake_logit))
    generator_loss = tf.reduce_mean(generator_loss)
    self.log_vars.append(("g_loss_fake", generator_loss))

    return discriminator_loss, generator_loss


def hr_compute_losses(self, images, wrong_images, fake_images, embeddings):
    '''
    In: hr_(real_images, wrong_images, fake_images)
    Out: hr_(discriminator_loss, generator_loss)
    '''
    real_logit = self.model.hr_get_discriminator(images, embeddings)
    wrong_logit = self.model.hr_get_discriminator(wrong_images, embeddings)
    fake_logit = self.model.hr_get_discriminator(fake_images, embeddings)

    real_d_loss = tf.nn.sigmoid_cross_entropy_with_logits(real_logit, tf.ones_like(real_logit))
    real_d_loss = tf.reduce_mean(real_d_loss)
    wrong_d_loss = tf.nn.sigmoid_cross_entropy_with_logits(wrong_logit, tf.zeros_like(wrong_logit))
    wrong_d_loss = tf.reduce_mean(wrong_d_loss)
    fake_d_loss = tf.nn.sigmoid_cross_entropy_with_logits(fake_logit, tf.zeros_like(fake_logit))
    fake_d_loss = tf.reduce_mean(fake_d_loss)
    
    if cfg.TRAIN.B_WRONG:
        discriminator_loss = real_d_loss + (wrong_d_loss + fake_d_loss) / 2.
        self.log_vars.append(("hr_d_loss_wrong", wrong_d_loss))
    else:
        discriminator_loss = real_d_loss + fake_d_loss
    self.log_vars.append(("hr_d_loss_real", real_d_loss))
    self.log_vars.append(("hr_d_loss_fake", fake_d_loss))

    generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(fake_logit, tf.ones_like(fake_logit))
    generator_loss = tf.reduce_mean(generator_loss)
    self.log_vars.append(("hr_g_loss_fake", generator_loss))

    return discriminator_loss, generator_loss
