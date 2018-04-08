import math
import tensorflow as tf
import tensorflow.contrib.slim as slim

def ACGANGenerator(batch_size,
                   num_classes,
                   z_dim=128,
                   output_dim=256,
                   data_format='NCHW'):
    """
    Create AC-GAN generator graph

    Params:
        batch_size   (int). The batch size
        num_classes  (int). The number of classes
        z_dim        (int=128): The dimension of the input noise vector, z [128].
        output_dim   (int=256): The dimension of the image output (i.e. height and width) [256].
        data_format  (str='NCHW'): The data format to use for the image.

    Returns:
        Tensor[NCHW | NHWC]: A tensor representing the output image, with the same format as the data_format param.
        Tensor[None, num_classes]: A placeholder tensor representing the intended class labels for the batch.
    """
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('input'):
            z      = tf.random_normal([batch_size, z_dim], name         ='z')
            labels = tf.placeholder([None, num_classes], tf.int32, name ='labels')
            noise  = tf.concat([noise, labels], 1)

        """
        [None, 128] -> [4, 4, 8*output_dim]
        """
        with tf.variable_scope('layers'):
            output = slim.fully_connected(noise, 4*4*8*output_dim*2)
            output = tf.reshape(output, [-1, 8*output_dim*2, 4, 4])
            if batch_norm:
                output = slim.batch_norm(output)

            output = pixcnn_gated_nonlinearity(output, labels, biases=False)

            num_upscales = math.log(output_dim, 2) - 2
            """
            dimension: 4    -> 8    -> 16  -> 32  -> 64 -> 128 -> 256
            filters:   2048 -> 1024 -> 512 -> 256 -> 128 -> 64 -> 32
            """
            for i in range(num_upscales):
                with tf.variable_scope(str(i)):
                    dimension = 2**(i+2)
                    filters   = tf.shape(output)[-1] / 2
                    output = upscale(output,
                                     [-1, dimension, dimension, filters],
                                     data_format=data_format,
                                     batch_norm=True)
                    output = pixcnn_gated_nonlinearity(output, labels)

        with tf.variable_scope('output'):
            output = upsample(output, [-1, output_dim, output_dim, 3])
            output = tf.nn.tanh(output)

            return output, labels

def upscale(input_tensor,
            output_shape,
            kernel=5,
            stride=2,
            stddev=0.02,
            data_format='NCHW',
            batch_norm=True):
    with tf.variable_scope('upscale'):
        resize_shape = [
            (output_shape[1] - 1) * stride + (kernel - 4),
            (output_shape[2] - 1) * stride + (kernel - 4)
        ]

        if data_format == 'NCHW':
            input_tensor = tf.transpose(input_tensor, [0, 2, 3, 1]) # Convert input to NHWC for resizing
        resized = tf.image.resize_nearest_neighbor(input_tensor, resize_shape)
        if data_format == 'NCHW':
            resized  = tf.transpose(resized, [0, 3, 1, 2]) # Convert back to NCHW
        filters = output_shape[-1]

        output = slim.conv2d(resized,
                    filters,
                    kernel_size=kernel,
                    stride=stride,
                    padding='SAME',
                    data_format=data_format,
                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                    biases_initializer=tf.constant_initializer(0.0))

        if batch_norm:
            output = slim.batch_norm(output, data_format=data_format)

        return output

def pixcnn_gated_nonlinearity(input_tensor, labels, biases=True):
    with tf.variable_scope('pixcnn_gated_nonlinearity'):
        condition      = generate_condition(input_tensor, labels, biases=biases)

        even_input     = input_tensor[:, ::2]
        even_condition = condition_tensor[:, ::2]
        odd_input      = input_tensor[:,1::2]
        odd_condition  = condition_tensor[:,1::2]

        even_tensor    = even_input + even_condition
        odd_tensor     = odd_input + odd_condition

        return tf.mul(tf.sigmoid(even_tensor), tf.tanh(odd_tensor))

def generate_condition(input_tensor, labels, biases):
    with tf.variable_scope('condition'):
        input_shape = tf.shape(input_tensor)
        label_dim   = tf.reduce_prod(input_shape[1:])
        output      = slim.fully_connected(labels, label_dim)
        output      = tf.reshape(output, input_shape)
        return output
