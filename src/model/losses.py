import tensorflow as tf
import tensorflow.contrib.slim as slim

def class_accuracy(prediction, labels):
    with tf.variable_scope('accuracy'):
        prediction_index = tf.argmax(prediction, 1)
        label_index      = tf.argmax(labels, 1)
        is_class_correct = tf.equal(prediction_index, label_index)
        accuracy         = tf.reduce_mean(tf.cast(is_class_correct, tf.float32))
        return accuracy

def class_loss(prediction, labels):
    with tf.variable_scope('class_loss'):
        accuracy = class_accuracy(prediction, labels)

        with tf.variable_scope('softmax'):
            softmax = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels))

        return softmax, accuracy

def interpolates(real_batch, fake_batch, alpha):
    with tf.variable_scope('interpolates'):
        real_batch_flat = slim.flatten(real_batch)
        fake_batch_flat = slim.flatten(fake_batch)
        differences  = fake_batch_flat - real_batch_flat
        interpolates = real_batch_flat + (alpha*differences)
        return tf.reshape(interpolates, tf.shape(real_batch))

def lambda_gradient_penalty(disc, interpolates, lambda_penalty):
    with tf.variable_scope('lambda_gradient_penalty'):
        disc_flat = tf.reshape(disc, [-1])
        gradients = tf.gradients(disc_flat, [interpolates])

        print(gradients)

        gradients = gradients[0]

        slopes    = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)

        return lambda_penalty*gradient_penalty

def total_gen_loss(disc_result, class_loss):
    with tf.variable_scope('total_loss'):
        gen_cost  = -tf.reduce_mean(disc_result)
        gen_cost += class_loss
        return gen_cost

def total_disc_loss(real_disc, fake_disc, class_loss, lambda_gradient_penalty):
    with tf.variable_scope('total_loss'):
        disc_cost   = tf.reduce_mean(fake_disc) - tf.reduce_mean(real_disc)
        disc_cost  += class_loss
        disc_cost  += lambda_gradient_penalty

        return disc_cost
