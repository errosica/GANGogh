import math
import tensorflow as tf
import tensorflow.contrib.slim as slim

class ACGANDiscriminator():
    def __init__(self, num_classes, image_size=256, data_format='NCHW'):
        """
        Initialize AC-GAN Discriminator

        Params:
            num_classes  (int): The number of classes.
            data_format  (str='NCHW'): The data format to use for the image.
            output_dim   (int=256): The dimension of the image output (i.e. height and width) [256].
        """
        self.num_classes = num_classes
        self.data_format = data_format
        self.image_size  = image_size

    def build_graph(self, output):
        """
        Build AC-GAN Discriminator graph.

        Params:
            input_tensor (Tensor[NCHW | NHWC]). An image tensor tensor representing the class label.

        Returns:
            Tensor[NCHW | NHWC]: A tensor representing the output image, with the same format as the data_format param.
            Tensor[None, num_classes]: A tensor representing the classification for the image.
        """
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            num_convolutions = int(math.log(self.image_size, 2) - 1)

            with tf.variable_scope('layers'):
                for i in range(num_convolutions):
                    with tf.variable_scope(str(i)):
                        dim = self.image_size*(2**i)
                        output = self.conv2d(output, dim, batch_norm=i > 0)

            with tf.variable_scope('output'):
                output        = slim.flatten(output)
                source_output = slim.fully_connected(output, 1)
                class_output  = slim.fully_connected(output, self.num_classes)

                return source_output, class_output

    def conv2d(self,
               input_tensor,
               output_dim,
               kernel_size=5,
               stride=2,
               stddev=0.02,
               batch_norm=True):
        output = slim.conv2d(input_tensor,
                             output_dim,
                             kernel_size,
                             stride=stride,
                             data_format=self.data_format,
                             weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                             biases_initializer=tf.constant_initializer(0.0))
        if batch_norm:
            output = slim.batch_norm(output, data_format=self.data_format)
        return tf.nn.leaky_relu(output)
