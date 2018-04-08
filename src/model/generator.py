import math
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

class ACGANGenerator():
    def __init__(self, num_classes, image_size=256, data_format='NCHW'):
        """
        Initialize AC-GAN generator

        Params:
            num_classes  (int). The number of classes
            image_size   (int=256): The dimension of the image output (i.e. height and width) [256].
            data_format  (str='NCHW'): The data format to use for the image.
        """
        self.num_classes = num_classes
        self.image_size = image_size
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
                output    = slim.fully_connected(z, 4*4*8*self.image_size*2)
                output    = tf.reshape(output, [-1, 8*self.image_size*2, 4, 4])
                output    = slim.batch_norm(output)
                condition = self.generate_condition(output, labels, biases=False)
                output    = self.pixcnn_gated_nonlinearity(output, condition)

                num_upsamples = int(math.log(self.image_size, 2) - 2)
                """
                dimension: 4    -> 8    -> 16  -> 32  -> 64 -> 128 -> 256
                filters:   2048 -> 1024 -> 512 -> 256 -> 128 -> 64 -> 32
                """
                for i in range(num_upsamples):
                    with tf.variable_scope(str(i)):
                        dimension = 2**(i+2)
                        filters   = (4/(2**i))*(self.image_size*2)
                        output = self.upsample(output,
                                         [-1, dimension, dimension, filters],
                                         batch_norm=True)
                        print(output, output.shape)
                        condition = self.generate_condition(output, labels)
                        output    = self.pixcnn_gated_nonlinearity(output, condition)

            with tf.variable_scope('output'):
                output = self.upsample(output, [-1, self.image_size, self.image_size, 3])
                output = tf.nn.tanh(output)

                return output

    def upsample(self,
                 input_tensor,
                 output_shape,
                 kernel=5,
                 stride=2,
                 stddev=0.02,
                 batch_norm=False):
        with tf.variable_scope('upsample'):
            resize_shape = [
                (output_shape[1] - 1) * stride + (kernel - 4),
                (output_shape[2] - 1) * stride + (kernel - 4)
            ]

            if self.data_format == 'NCHW':
                input_tensor = tf.transpose(input_tensor, [0, 2, 3, 1]) # Convert input to NHWC for resizing
            resized = tf.image.resize_nearest_neighbor(input_tensor, resize_shape)
            if self.data_format == 'NCHW':
                resized  = tf.transpose(resized, [0, 3, 1, 2]) # Convert back to NCHW
            filters = output_shape[-1]

            output = slim.conv2d(resized,
                        filters,
                        kernel_size=kernel,
                        stride=stride,
                        padding='SAME',
                        data_format=self.data_format,
                        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                        biases_initializer=tf.constant_initializer(0.0))

            if batch_norm:
                output = slim.batch_norm(output, data_format=self.data_format)

            return output

    def pixcnn_gated_nonlinearity(self, input_tensor, condition_tensor):
        with tf.variable_scope('pixcnn_gated_nonlinearity'):
            even_input     = input_tensor[:, ::2]
            even_condition = condition_tensor[:, ::2]
            odd_input      = input_tensor[:,1::2]
            odd_condition  = condition_tensor[:,1::2]

            even_tensor    = even_input + even_condition
            odd_tensor     = odd_input + odd_condition

            return tf.sigmoid(even_tensor) * tf.tanh(odd_tensor)

    def generate_condition(self, input_tensor, labels, biases=True):
        with tf.variable_scope('condition'):
            flat_shape = int(np.prod(input_tensor.get_shape()[1:]))

            if biases:
                output = slim.fully_connected(labels, flat_shape)
            else:
                output = slim.fully_connected(labels, flat_shape, biases_initializer=None)
            output = tf.reshape(output, tf.shape(input_tensor))
            return output
