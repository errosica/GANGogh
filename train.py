from src.dataset import get_dataset_iterator
from src.model.discriminator import ACGANDiscriminator as Discriminator
from src.model.generator import ACGANGenerator as Generator
import src.model.losses as losses
import tensorflow as tf
import numpy as np
from config import config

log_dir             = config.log_dir
data_dir            = config.data_dir
batch_size          = config.batch_size
z_dim               = config.z_dim
num_epochs          = config.num_epochs
image_size          = config.image_size
lambda_penalty      = config.lambda_penalty
num_classes         = config.num_classes
data_format         = config.data_format
gen_lr              = config.gen_lr
disc_lr             = config.disc_lr
disc_iters          = config.disc_iters
disc_lr_decay_rate  = config.disc_lr_decay_rate
disc_lr_decay_steps = config.disc_lr_decay_steps # TODO: scale decay steps for disc lr by the disc_iters
gen_lr_decay_rate   = config.gen_lr_decay_rate
gen_lr_decay_steps  = config.gen_lr_decay_steps

generator_instance     = Generator(num_classes, image_size, data_format)
discriminator_instance = Discriminator(num_classes, image_size, data_format)

def create_train_op(loss_tensor, learning_rate, global_step, var_list, decay_steps, decay_rate):
    exp_learning_rate = tf.train.exponential_decay(learning_rate,
                                                   global_step,
                                                   decay_steps=decay_steps,
                                                   decay_rate=decay_rate)

    optimizer = tf.train.AdamOptimizer(learning_rate=exp_learning_rate, beta1=0.5, beta2=0.9)

    return optimizer.minimize(loss_tensor, var_list=var_list, global_step=global_step, colocate_gradients_with_ops=True), exp_learning_rate

def random_labels():
    return np.eye(num_classes)[np.random.choice(num_classes, batch_size)]

def random_z():
    return np.random.normal(0.0, 1.0, [batch_size, z_dim])

def random_feed_dict(z, labels):
    """
    Generate a random z and label pair as input into sess.run()
    """
    return {
        z: random_z(),
        labels: random_labels()
    }

"""
Run training routine
"""
def main(_):
    print("Training!")
    global_step = tf.train.get_or_create_global_step()

    with tf.variable_scope('dataset'):
        iterator = get_dataset_iterator(data_dir,
                                        batch_size,
                                        scale_size=image_size,
                                        num_classes=num_classes,
                                        data_format=data_format)

        real_batch, real_labels = iterator.get_next()

    with tf.variable_scope('inputs'):
        z      = tf.placeholder(tf.float32, shape=[None, z_dim], name='z')
        labels = tf.placeholder(tf.float32, shape=[None, num_classes], name='labels')
        fake_labels = tf.identity(labels)

    fake_batch  = generator_instance.build_graph(z, labels)
    real_disc, real_class_disc = discriminator_instance.build_graph(real_batch)
    fake_disc, fake_class_disc = discriminator_instance.build_graph(fake_batch)

    gen_summaries  = []
    disc_summaries = []

    with tf.variable_scope('losses'):
        with tf.variable_scope('gen'):
            gen_class_loss, gen_class_accuracy = losses.class_loss(fake_class_disc, fake_labels)
            total_gen_loss = losses.total_gen_loss(fake_disc, gen_class_loss)

            gen_summaries.append(tf.summary.scalar('class_loss', gen_class_loss))
            gen_summaries.append(tf.summary.scalar('class_accuracy', gen_class_accuracy))
            gen_summaries.append(tf.summary.scalar('total_loss', total_gen_loss))

        with tf.variable_scope('disc'):
            disc_class_loss, disc_class_accuracy = losses.class_loss(real_class_disc, real_labels)

            alpha = tf.random_uniform([batch_size, 1], minval=0., maxval=1., name='alpha')

            interpolates = losses.interpolates(real_batch, fake_batch, alpha)
            interp_disc, interp_class_disc = discriminator_instance.build_graph(interpolates)

            lambda_gradient_penalty = losses.lambda_gradient_penalty(interp_disc, interpolates, lambda_penalty=lambda_penalty)

            total_disc_loss = losses.total_disc_loss(real_disc, fake_disc, disc_class_loss, lambda_gradient_penalty)

            disc_summaries.append(tf.summary.scalar('class_loss', disc_class_loss))
            disc_summaries.append(tf.summary.scalar('class_accuracy', disc_class_accuracy))
            disc_summaries.append(tf.summary.scalar('lambda_gradient_penalty', lambda_gradient_penalty))
            disc_summaries.append(tf.summary.scalar('total_loss', total_disc_loss))

    with tf.variable_scope('optimizers'):
        trainable_vars = tf.trainable_variables()
        gen_vars  = [var for var in trainable_vars if 'generator' in var.name]
        disc_vars = [var for var in trainable_vars if 'discriminator' in var.name]

        gen_train_op, gen_exp_lr = create_train_op(total_gen_loss,
                                         learning_rate=gen_lr,
                                         global_step=global_step,
                                         var_list=gen_vars,
                                         decay_rate=gen_lr_decay_rate,
                                         decay_steps=gen_lr_decay_steps)
        disc_train_op, disc_exp_lr = create_train_op(total_disc_loss,
                                         learning_rate=disc_lr,
                                         global_step=global_step,
                                         var_list=disc_vars,
                                         decay_rate=disc_lr_decay_rate,
                                         decay_steps=disc_lr_decay_steps)


    disc_summaries.append(tf.summary.image('real', real_batch, max_outputs=10))
    gen_summaries.append(tf.summary.image('generated', fake_batch, max_outputs=10))

    gen_summaries.append(tf.summary.scalar('gen_global_step', gen_global_step))
    gen_summaries.append(tf.summary.scalar('gen_learning_rate', gen_exp_lr))
    disc_summaries.append(tf.summary.scalar('disc_global_step', disc_global_step))
    gen_summaries.append(tf.summary.scalar('disc_learning_rate', disc_exp_lr))

    gen_summaries  = tf.summary.merge(gen_summaries)
    disc_summaries = tf.summary.merge(disc_summaries)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement     = True
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=log_dir,
        save_summaries_steps=1,
        save_checkpoint_secs=300,
        config=sess_config
    ) as sess:
        """
        First pre-train the discriminator for 100 steps
        """
        print('Starting pretraining routine')
        # for i in range(100):
        #     sess.run([disc_train_op, disc_summaries], feed_dict=random_feed_dict(z, labels))
        print('Pretraining completed')

        """
        Run N training epochs
        """
        for n in range(num_epochs):
            epoch_num = n+1
            print('Starting epoch %d' % epoch_num)
            while True:
                try:
                    sess.run([gen_train_op, gen_summaries], feed_dict=random_feed_dict(z, labels))
                    """
                    Run M discriminator training iterators for every generator iteration
                    """
                    for i in range(disc_iters):
                        sess.run([disc_train_op, disc_summaries], feed_dict=random_feed_dict(z, labels))
                except tf.errors.OutOfRangeError:
                    print('Finished epoch %d' % epoch_num)
                    break
        print('Training finished')

if __name__ == '__main__':
    tf.app.run()
