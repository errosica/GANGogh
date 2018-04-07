import src.pipeline as pipeline
from src.model.discriminator import ACGANDiscriminator as Discriminator
from src.model.generator import ACGANGenerator as Generator
import src.model.losses as losses
import tensorflow as tf
import numpy as np
from config import config

log_dir           = config.log_dir
data_dir          = config.data_dir
batch_size        = config.batch_size
z_dim             = config.z_dim
image_output_size = config.image_output_size
lambda_penalty    = config.lambda_penalty
num_classes       = config.num_classes
data_format       = config.data_format
gen_lr            = config.gen_lr
disc_lr           = config.disc_lr
disc_iters        = config.disc_iters

def create_train_op(loss_tensor, learning_rate, global_step, var_list):
    exp_learning_rate = tf.train.exponential_decay(learning_rate,
                                                   global_step,
                                                   decay_steps=100000,
                                                   decay_rate=0.96)

    optimizer = tf.train.AdamOptimizer(learning_rate=exp_learning_rate,
                                       global_step=global_step,
                                       beta1=0.5, beta2=0.9)

    return optimizer.minimize(loss_tensor, var_list=var_list, colocate_gradients_with_ops=True)

def random_labels(batch_size, num_classes):
    return np.random.choice(np.eye(num_classes), batch_size)

"""
Run training routine
"""
def main():
    gen_global_step  = tf.Variable(0, trainable=False, name='gen_global_step')
    disc_global_step = tf.Variable(0, trainable=False, name='disc_global_step')

    iterator = pipeline.get_dataset(data_dir,
                                    batch_size,
                                    scale_size=image_output_size,
                                    num_classes=num_classes,
                                    data_format=data_format)

    real_batch, real_labels = iterator.get_next()

    fake_batch, fake_labels = Generator(batch_size,
                                        num_classes,
                                        z_dim,
                                        output_dim=image_output_size,
                                        data_format=data_format)

    real_disc, real_class_disc = Discriminator(real_batch, num_classes=num_classes)
    fake_disc, fake_class_disc = Discriminator(fake_batch, num_classes=num_classes)

    gen_cost, disc_cost, class_cost, gen_summaries, disc_summaries = losses.calculate(batch_size, num_classes, lambda_penalty,
                                                                                      real_disc, real_class_disc, real_labels,
                                                                                      fake_disc, fake_class_disc, fake_labels)

    disc_summaries.append(tf.summary.image('Real', real_batch, max_outputs=10))
    gen_summaries.append(tf.summary.image('Generated', fake_batch, max_outputs=10))

    gen_summaries  = tf.summaries.merge(gen_summaries)
    disc_summaries = tf.summaries.merge(disc_summaries)

    with tf.variable_scope('optimizers'):
        trainable_vars = tf.trainable_variables()
        gen_vars  = [var for var in trainable_vars if 'Generator' in var.name]
        disc_vars = [var for var in trainable_vars if 'Discriminator' in var.name]

        gen_train_op   = create_train_op(gen_cost, learning_rate=gen_lr, global_step=gen_global_step, var_list=gen_vars)
        disc_train_op  = create_train_op(disc_cost, learning_rate=disc_lr, global_step=disc_global_step, var_list=disc_vars)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement     = True
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=log_dir,
        save_checkpoint_secs=300,
        config=sess_config
    ) as sess:
        """
        First pre-train the discriminator for 100 steps
        """
        print('Starting pretraining routine')
        sess.run(iterator.initialize)
        for i in range(100):
            sess.run([disc_train_op, disc_summaries], feed_dict={
                gen_labels: random_labels(batch_size, num_classes)
            })
            tf.train.global_step(sess, disc_global_step)
        print('Pretraining completed')

        """
        Run N training epochs
        """
        for n in range(num_epochs):
            print('Starting epoch %d' % n + 1)
            sess.run(iterator.initialize)
            while True:
                try:
                    sess.run([gen_train_op, gen_summaries], feed_dict={ gen_labels: random_labels(batch_size, num_classes) })
                    tf.train.global_step(sess, gen_global_step)

                    """
                    Run 5 discriminator training iterators for every generator iteration
                    """
                    for i in range(disc_iters):
                        sess.run([disc_train_op, disc_summaries], feed_dict={ gen_labels: random_labels(batch_size, num_classes) })
                        tf.train.global_step(sess, disc_global_step)
                except tf.errors.OutOfRangeError:
                    print('Finished epoch %d' % n + 1)
                    break

        print('Training finished')
