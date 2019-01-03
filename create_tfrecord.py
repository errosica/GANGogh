import argparse, os

# python ./create_tfrecord.py F:/Data/smallimages1024_test
parser = argparse.ArgumentParser(description='Create Tensorflow Records')

parser.add_argument('Input', action="store", help="The path to the images", type=str)
parser.add_argument('--Output', action="store", help="The path to save the tensorflow records. Defaults to Input path", type=str)
parser.add_argument('--Max_Size', action="store", default="10000000000",help="The maximum size of the tensorflow records. Default is 10GB", type=int)
p=parser.parse_args()

INPUT = p.Input 
default = os.path.join(INPUT,r"default.tfrecords")
OUTPUT = os.path.join(p.Output, r"default.tfrecords") if p.Output else default
max_shard_size = p.Max_Size
print("Taking input from folder: ", INPUT)
print("Writing output to folder: ", OUTPUT)

import cv2, glob, tqdm, tensorflow as tf  #only import these if the arguments are well formatted
from tqdm import tqdm

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


writer = tf.python_io.TFRecordWriter(OUTPUT)
shard_counter = 1
estimated_shard_size = 0
ROOT = r"F:/Data/smallimages1024_test/"
globstr = ROOT + '*/**.png'
files = glob.glob(globstr, recursive=True)
for number, f in tqdm(enumerate(files), total = len(files)):
  #print("Processing file: ", f)
  if number == 0:                                           #estimate that every file is the same size as the first
    estimated_size = os.path.getsize(f)
  img = cv2.imread(f)
  height, width, channels = img.shape
  folder = os.path.split(os.path.split(files[7])[0])[1]
  example = tf.train.Example(
    features = tf.train.Features(
      feature = {
        "image": _bytes_feature(img.tostring()),
        "height": _int64_feature(height),
        "width": _int64_feature(width),
        "channels": _int64_feature(channels),
        "label": _int64_feature(number)
      }
    )
  )
  writer.write(example.SerializeToString())
  estimated_shard_size+=estimated_size
  if estimated_shard_size > max_shard_size:
    writer.close()
    NEW_OUTPUT = OUTPUT.split('.')[0] + str(shard_counter) + '.' + OUTPUT.split('.')[1]
    #print("Max Shard Size Exceeded. Writing to: ", NEW_OUTPUT)
    writer = tf.python_io.TFRecordWriter(NEW_OUTPUT)
    estimated_shard_size = 0
    shard_counter +=1
   
writer.close()