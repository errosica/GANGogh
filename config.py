import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string("data_dir", "./data", "The data directory. Can be a google storage path (e.g. gs://my-bucket/my/path)")
flags.DEFINE_string("log_dir", "./logs", "Directory to store logs. Can be a google storage path (e.g. gs://my-bucket/my/path)")

flags.DEFINE_integer("batch_size", 16, "The batch size [16]")
flags.DEFINE_integer("z_dim", 128, "The z input dimension [128]")
flags.DEFINE_integer("image_output_size", 256, "The width and height of the output images [256]")
flags.DEFINE_integer("num_classes", 9, "The number of classes that exist in the training set [9]")
flags.DEFINE_integer("lambda_penalty", 10, "The lambda gradient penalty [10]")
flags.DEFINE_string("data_format", "NHWC", "The image data format to use. NCHW is recommended for GPU computations. [NHWC]")

flags.DEFINE_float("gen_lr", 1e-5, "The generator learning rate [1e-5]")
flags.DEFINE_float("disc_lr", 1e-4, "The discriminator learning rate [1e-4]")

flags.DEFINE_integer("disc_iters", 5, "The number of discriminator training iterations to run per training step [5].")

config = flags.FLAGS
