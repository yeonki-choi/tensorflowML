import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
tf.__version__

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([784])
    image = tf.cast(image, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.int32)

    return image, label

def input_fn(filename, batch_size=100):
    filename_queue = tf.train.string_input_producer([filename])

    image, label = read_and_decode(filename_queue)
    return image,label


sess = tf.Session()
sess.run(tf.global_variables_initializer())

i,l = input_fn('/tmp/data/validation.tfrecords')
print ('read data')
v_image,v_label = sess.run([i,l])
print ('value',v_image,v_label)

