import math
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

class ACGANGenerator():
    def __init__(self, num_classes, image_size, data_format):
        """
        Initialize AC-GAN generator

        Params:
            num_classes  (int). The number of classes
            image_size   (int=256): The dimension of the image output (i.e. height and width) [256].
            data_format  (str='NCHW'): The data format to use for the image.
        """
        self.num_classes = num_classes
        self.image_size  = image_size
        self.data_format = data_format

    def build_graph(self, z, labels):
        """
        Build AC-GAN generator graph

        Params:
            z       (Tensor[None, None]): A 2-D tensor representing the z-input.
            labels  (Tensor[None, None]): A 2-D tensor representing the labels for each z-input.

        Returns:
            Tensor[NCHW | NHWC]: A tensor representing the output image, with the same format as the data_format param.
            Tensor[None, num_classes]: A placeholder tensor representing the intended class labels for the batch.
        """
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            z = tf.concat([z, labels], 1)

            """
            [None, 128] -> [4, 4, 8*output_dim]
            """
            with tf.variable_scope('layers'):
                output    = slim.fully_connected(z, 4*4*8*self.image_size*2, activation_fn=None)

                initial_output_shape = [-1, 4, 4, 8*self.image_size*2] if self.data_format == 'NHWC' else [-1, 8*self.image_size*2, 4, 4]
                output    = tf.reshape(output, initial_output_shape)
                output    = slim.batch_norm(output, data_format=self.data_format)
                condition = self.generate_condition(output, labels, biases=False)
                output    = self.pixcnn_gated_nonlinearity(output, condition)

                num_upsamples = int(math.log(self.image_size, 2) - 2)
                """
                dimension: 4    -> 8    -> 16  -> 32  -> 64 -> 128 -> 256
                filters:   2048 -> 1024 -> 512 -> 256 -> 128 -> 64 -> 32
                """
                for i in range(num_upsamples):
                    with tf.variable_scope(str(i)):
                        dimension = int(2**(i+2))
                        filters   = int(4/(2**i)*(self.image_size*2))
                        output    = self.upsample(output,
                                                  [-1, dimension, dimension, filters],
                                                  batch_norm=True)
                        condition = self.generate_condition(output, labels)
                        output    = self.pixcnn_gated_nonlinearity(output, condition)

            with tf.variable_scope('output'):
                output = self.upsample(output, [-1, self.image_size, self.image_size, 3])
                output = tf.nn.tanh(output)

                return output

    def upsample(self,
                 input_tensor,
                 output_shape,
                 kernel_size=5,
                 stride=2,
                 batch_norm=True):

        filters = output_shape[-1]
        height  = output_shape[1]
        width   = output_shape[2]

        with tf.variable_scope('upsample_%d_%d_%d' % (height, width, filters)):
            output  = tf.image.resize_nearest_neighbor(
                input_tensor,
                ((height-1)*stride + kernel_size-4,(width-1)*stride + kernel_size-4)
            )

            output = slim.conv2d(output, filters, kernel_size, stride=stride, data_format=self.data_format)

            if batch_norm:
                output = slim.batch_norm(output, data_format=self.data_format)

            return output

    def pixcnn_gated_nonlinearity(self, input_tensor, condition_tensor):
        with tf.variable_scope('pixcnn_gated_nonlinearity'):
            if self.data_format == 'NCHW':
                input_tensor = tf.transpose(input_tensor, [0, 2, 3, 1])
                condition_tensor = tf.transpose(condition_tensor, [0, 2, 3, 1])

            even_input     = input_tensor[:, ::2]
            even_condition = condition_tensor[:, ::2]
            odd_input      = input_tensor[:,1::2]
            odd_condition  = condition_tensor[:,1::2]

            even_tensor    = even_input + even_condition
            odd_tensor     = odd_input + odd_condition

            output = tf.sigmoid(even_tensor) * tf.tanh(odd_tensor)

            if self.data_format == 'NCHW':
                output = tf.transpose(output, [0, 3, 1, 2])

            return output

    def generate_condition(self, input_tensor, labels, biases=True):
        with tf.variable_scope('condition'):
            flat_shape = int(np.prod(input_tensor.get_shape()[1:]))

            if biases:
                output = slim.fully_connected(labels, flat_shape, activation_fn=None)
            else:
                output = slim.fully_connected(labels, flat_shape, activation_fn=None, biases_initializer=None)
            output = tf.reshape(output, tf.shape(input_tensor))
            return output
