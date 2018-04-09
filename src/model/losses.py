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
        differences  = fake_batch - real_batch
        interpolates = real_batch + (alpha*differences)
        return interpolates

def lambda_gradient_penalty(interp_disc, interpolates, lambda_penalty):
    with tf.variable_scope('lambda_gradient_penalty'):
        gradients = tf.gradients(interp_disc, [interpolates])
        gradients = gradients[0]
        slopes    = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)

        return lambda_penalty*gradient_penalty

def total_gen_loss(mean_fake_disc, class_loss):
    with tf.variable_scope('total_loss'):
        return class_loss - mean_fake_disc

def total_disc_loss(mean_real_disc, mean_fake_disc, class_loss, lambda_gradient_penalty):
    with tf.variable_scope('total_loss'):
        return mean_fake_disc - mean_real_disc + class_loss + lambda_gradient_penalty
