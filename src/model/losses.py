import tensorflow as tf
from src.model.discriminator import ACGANDiscriminator as Discriminator

def calculate(batch_size,
              num_classes,
              lambda_penalty,
              real_disc,
              real_class_disc,
              real_labels,
              fake_disc,
              fake_class_disc,
              fake_labels):
    with tf.variable_scope('losses'):

        gen_summaries  = []
        disc_summaries = []

        with tf.variable_scope('real_fake'):
            with tf.variable_scope('generated'):
                fake_class_prediction = tf.argmax(fake_class_disc, 1)
                fake_labels           = tf.argmax(fake_labels, 1)
                is_fake_class_correct = tf.equal(fake_class_prediction, fake_labels)
                accuracy_fake_class   = tf.reduce_mean(tf.cast(is_fake_class_correct, tf.float32))

            with tf.variable_scope('real'):
                real_class_prediction = tf.argmax(real_class_disc, 1)
                real_labels           = tf.argmax(real_labels, 1)
                is_real_class_correct = tf.equal(real_class_prediction, real_labels)
                accuracy_real_class   = tf.reduce_mean(tf.cast(is_real_class_correct, tf.float32))

        with tf.variable_scope('classes'):
            with tf.variable_scope('generated'):
                fake_class_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fake_class_disc, labels=fake_labels))

            with tf.variable_scope('real'):
                real_class_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=real_class_disc, labels=real_labels))

        with tf.variable_scope('interpolates'):
            alpha = tf.random_uniform(
                shape=[batch_size, 1],
                minval=0.,
                maxval=1.,
                name='alpha'
            )

            differences  = fake_data - real_data
            interpolates = real_data + (alpha*differences)

            interp_disc, interp_class_disc = Discriminator(interpolates, num_classes=num_classes)
            gradients = tf.gradients(interp_disc, [interpolates])[0]
            slopes    = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)

            lambda_gradient_penalty = lambda_penalty*gradient_penalty

        with tf.variable_scope('total'):
            with tf.variable_scope('generator'):
                gen_cost  = -tf.reduce_mean(fake_disc)
                gen_cost += fake_class_cost

            with tf.variable_scope('discriminator'):
                disc_cost   = tf.reduce_mean(fake_disc) - tf.reduce_mean(real_disc)
                disc_cost  += real_class_cost
                disc_cost  += lambda_gradient_penalty

    disc_summaries.append(tf.summary.scalar('disc/accuracy_class_fake', accuracy_fake_class))
    disc_summaries.append(tf.summary.scalar('disc/accuracy_class_real', accuracy_real_class))
    disc_summaries.append(tf.summary.scalar('disc/class_cost_fake', fake_class_cost))
    disc_summaries.append(tf.summary.scalar('disc/class_cost_real', real_class_cost))
    disc_summaries.append(tf.summary.scalar('disc/total_cost', disc_cost))
    gen_summaries.append(tf.summary.scalar('gen/total_cost', gen_cost))

    return gen_cost, disc_cost, real_class_cost_gradient, gen_summaries, disc_summaries
