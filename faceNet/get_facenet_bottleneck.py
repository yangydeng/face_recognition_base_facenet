'''
@Date  : 2017-11-29 11:11
@Author: yangyang Deng
@Email : yangydeng@163.com
@Describe: 
    这里导入的模型来自github上的faceNet, 调用的模型为inception-resnet_v1
    URL: https://github.com/davidsandberg/facenet#pre-trained-model
'''


import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
MODEL_DIR = '/home/hiptonese/project/face_recognition/Facenet_pre_train_model/'
PB_FILENAME = '20170511-185253.pb'


def get_bottleneck(image_path_list, image_size=160):
    with gfile.FastGFile(MODEL_DIR + PB_FILENAME, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # bottleneck_name = 'InceptionResnetV1/Logits/Flatten/Reshape:0'
    bottleneck_name = 'embeddings:0'
    phase_name = 'phase_train:0'
    batch_size_name = 'batch_size:0'
    image_batch_name = 'image_batch:0'
    bottle_neck, phase, batch_size, image_batch = tf.import_graph_def(
        graph_def,
        return_elements=[bottleneck_name, phase_name, batch_size_name, image_batch_name]
    )

    processed_images = []
    for image_path in image_path_list:
        image_raw = tf.gfile.FastGFile(image_path, 'rb').read()
        image_decode = tf.image.decode_jpeg(image_raw, channels=3)
        image_data = tf.image.convert_image_dtype(image_decode, dtype=tf.float32)
        processed_image = tf.image.resize_images(image_data, [image_size, image_size], method=0)

        # image = tf.image.convert_image_dtype(processed_image, dtype=tf.uint8)
        # encode_image = tf.image.encode_jpeg(image)
        # with tf.Session() as sess:
        #     with tf.gfile.GFile(image_path+str(np.random.randint(0,1000)), 'wb') as f:
        #         f.write(encode_image.eval())

        processed_images.append(processed_image)

    with tf.Session() as sess:
        processed_images_eval = sess.run(processed_images)

        bottle_neck_eval = sess.run(bottle_neck,
                                    feed_dict={phase: False, batch_size: 16, image_batch: processed_images_eval})

        return bottle_neck_eval


def calculate_dis(b1, b2):
    sub = np.subtract(b1, b2)

    return np.sum(np.square(sub))


if __name__ == '__main__':
    image_list = []
    p1 = '/home/hiptonese/project/face_recognition/dataset/face/dyy.png'

    # image_list.append(p1)

    bottlenecks = get_bottleneck(image_list)
    # dyy = bottlenecks[0]

    # print('dyy - dyy1', calculate_dis(dyy, dyy1))



