import math
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

class ACGANDiscriminator():
    def __init__(self, num_classes, image_size, data_format):
        """
        Initialize AC-GAN Discriminator

        Params:
            num_classes  (int): The number of classes.
            data_format  (str): The data format to use for the image.
            output_dim   (int): The dimension of the image output (i.e. height and width) [256].
        """
        self.num_classes = num_classes
        self.data_format = data_format
        self.image_size  = image_size

    def build_graph(self, input_tensor):
        """
        Build AC-GAN Discriminator graph.

        Params:
            input_tensor (Tensor[NCHW | NHWC]). An image tensor tensor representing the class label.

        Returns:
            Tensor[None]: A 1-D tensor representing the probability (between 0 and 1) that each image in the batch is real.
            Tensor[None, num_classes]: A tensor representing the classification for each image in the batch.
        """
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            if self.data_format == 'NHWC':
                image_shape = [-1, self.image_size, self.image_size, 3]
            else:
                image_shape = [-1, 3, self.image_size, self.image_size]

            output = tf.reshape(input_tensor, image_shape)

            num_convolutions = int(math.log(self.image_size, 2) - 2)

            for i in range(num_convolutions):
                filters = int(self.image_size*(2**i))
                with tf.variable_scope('conv2d_%d' % filters):
                    output = slim.conv2d(output, filters, kernel_size=5, stride=2, data_format=self.data_format)
                    if i > 0:
                        output = slim.layer_norm(output)
                    output = tf.nn.leaky_relu(output)

            with tf.variable_scope('output'):
                output = slim.flatten(output)

                with tf.variable_scope('prob'):
                    source_output = self.dense(output, 1)
                    source_output = tf.reshape(source_output, [-1])
                    source_output = tf.Print(source_output, [source_output], 'Discriminator real/fake: ')

                with tf.variable_scope('classes'):
                    class_output  = self.dense(output, self.num_classes)
                    class_output  = tf.Print(class_output, [class_output], 'Discriminator classes: ')

                return source_output, class_output

    def dense(self, input_tensor, output_num):
        return slim.fully_connected(input_tensor, output_num, activation_fn=None)
