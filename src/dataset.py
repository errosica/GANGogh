import tensorflow as tf
import numpy as np

def get_dataset_iterator(data_dir,
                         batch_size,
                         scale_size,
                         num_classes,
                         buffer_size=10000,
                         data_format='NCHW'):
    """Construct a TF dataset from a remote source"""
    def transform(tfrecord_proto):
        return transform_tfrecord(tfrecord_proto,
                                  scale_size=scale_size,
                                  num_classes=num_classes,
                                  data_format=data_format)

    tf_dataset  = tf.data.TFRecordDataset(data_dir)
    tf_dataset  = tf_dataset.map(transform)
    tf_dataset  = tf_dataset.shuffle(buffer_size=buffer_size)
    tf_dataset  = tf_dataset.batch(batch_size)
    tf_iterator = tf_dataset.make_initializable_iterator()

    return tf_iterator

def decode_image(image_buff, height, width, scale_size, data_format):
    """
    Take a raw image byte string and decode to an image

    Params:
        image_buff (str): Image byte string
        height (int): The original image height
        width (int): The original image width
        scale_size (int): The output image height and width
        data_format (str): The output image data format (NCHW or NHCW)

    Return:
        Tensor[NCHW | NHCW]: A tensor of shaperepresenting the RGB image.
    """
    image_arr       = tf.decode_raw(image_buff, out_type=tf.uint8)
    image_decoded   = tf.reshape(image_arr, tf.stack([height, width, 3], axis=0))
    image_decoded   = tf.reverse(image_decoded, [-1]) # BGR to RGB
    image_decoded   = tf.expand_dims(image_decoded, axis=0)
    image_resized   = tf.image.resize_nearest_neighbor(image_decoded, [scale_size, scale_size])
    if data_format == 'NCHW':
        image_resized = tf.transpose(image_resized, [0, 3, 1, 2])
    image_resized   = tf.squeeze(image_resized)
    image_converted = tf.to_float(image_resized)

    return image_converted

def decode_class(label, num_classes):
    return tf.one_hot(label, num_classes, dtype=tf.float32)

def transform_tfrecord(tf_protobuf, scale_size, num_classes, data_format):
    """
    Decode the tfrecord protobuf into the image.

    Params:
        tf_protobuf (proto): A protobuf representing the data record.
        scale_size (int): The output image height and width.

    Returns:
        Tensor[output_height, output_width, 3]: A tensor representing the decoded image.
    """

    features = {
        "image": tf.FixedLenFeature((), tf.string, default_value=""),
        "height": tf.FixedLenFeature([], tf.int64),
        "width": tf.FixedLenFeature([], tf.int64),
        "label": tf.FixedLenFeature((), tf.int64, default_value=0)
    }
    parsed_features = tf.parse_single_example(tf_protobuf, features)

    decoded_image = decode_image(parsed_features["image"],
                                 height=parsed_features["height"],
                                 width=parsed_features["width"],
                                 scale_size=scale_size,
                                 data_format=data_format)

    decoded_class = decode_class(parsed_features["label"], num_classes)

    return decoded_image, decoded_class
