from src.dataset import get_dataset_iterator
from src.model.discriminator import ACGANDiscriminator as Discriminator
from src.model.generator import ACGANGenerator as Generator
import src.model.losses as losses
import tensorflow as tf
import numpy as np
from config import config

log_dir        = config.log_dir
data_dir       = config.data_dir
batch_size     = config.batch_size
z_dim          = config.z_dim
image_size     = config.image_size
lambda_penalty = config.lambda_penalty
num_classes    = config.num_classes
data_format    = config.data_format
gen_lr         = config.gen_lr
disc_lr        = config.disc_lr
disc_iters     = config.disc_iters

generator_instance     = Generator(num_classes, image_size, data_format)
discriminator_instance = Discriminator(num_classes, image_size, data_format)

def create_train_op(loss_tensor, learning_rate, global_step, var_list):
    exp_learning_rate = tf.train.exponential_decay(learning_rate,
                                                   global_step,
                                                   decay_steps=100000,
                                                   decay_rate=0.96)

    optimizer = tf.train.AdamOptimizer(learning_rate=exp_learning_rate, beta1=0.5, beta2=0.9)

    return optimizer.minimize(loss_tensor, var_list=var_list, colocate_gradients_with_ops=True)

def random_labels(batch_size, num_classes):
    return np.random.choice(np.eye(num_classes), batch_size)

def random_z(batch_size, z_dim):
    return np.random.normal([batch_size, z_dim])

"""
Run training routine
"""
def main(_):
    print("Training!")
    with tf.variable_scope('global_step'):
        global_step      = tf.Variable(0, trainable=False, name='global_step')
        gen_global_step  = tf.Variable(0, trainable=False, name='gen_global_step')
        disc_global_step = tf.Variable(0, trainable=False, name='disc_global_step')

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

    with tf.variable_scope('summaries'):
        disc_summaries.append(tf.summary.image('real', real_batch, max_outputs=10))
        gen_summaries.append(tf.summary.image('generated', fake_batch, max_outputs=10))

        gen_summaries  = tf.summary.merge(gen_summaries)
        disc_summaries = tf.summary.merge(disc_summaries)

    with tf.variable_scope('optimizers'):
        trainable_vars = tf.trainable_variables()
        gen_vars  = [var for var in trainable_vars if 'generator' in var.name]
        disc_vars = [var for var in trainable_vars if 'discriminator' in var.name]

        gen_train_op   = create_train_op(total_gen_loss, learning_rate=gen_lr, global_step=gen_global_step, var_list=gen_vars)
        disc_train_op  = create_train_op(total_disc_loss, learning_rate=disc_lr, global_step=disc_global_step, var_list=disc_vars)

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
                z: random_z(batch_size, z_dim),
                labels: random_labels(batch_size, num_classes)
            })
            tf.train.global_step(sess, disc_global_step)
            tf.train.global_step(sess, global_step)
        print('Pretraining completed')

        """
        Run N training epochs
        """
        for n in range(num_epochs):
            print('Starting epoch %d' % n + 1)
            sess.run(iterator.initialize)
            while True:
                try:
                    sess.run([gen_train_op, gen_summaries], feed_dict={
                        z: random_z(batch_size, z_dim),
                        labels: random_labels(batch_size, num_classes)
                    })
                    tf.train.global_step(sess, gen_global_step)

                    """
                    Run 5 discriminator training iterators for every generator iteration
                    """
                    for i in range(disc_iters):
                        sess.run([disc_train_op, disc_summaries], feed_dict={
                            z: random_z(batch_size, z_dim),
                            labels: random_labels(batch_size, num_classes)
                        })
                        tf.train.global_step(sess, disc_global_step)

                    tf.train.global_step(sess, global_step)
                except tf.errors.OutOfRangeError:
                    print('Finished epoch %d' % n + 1)
                    break

        print('Training finished')

if __name__ == '__main__':
    tf.app.run()
