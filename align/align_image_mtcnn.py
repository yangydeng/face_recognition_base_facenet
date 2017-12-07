"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
from time import sleep
import sys

import imageio
import numpy as np
import tensorflow as tf
from scipy import misc
from skimage import transform

from align import detect_face
from faceNet import facenet


def align_image_mtcnn(args):
    sleep(random.random())
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cls = facenet.get_single_dataset(args.class_name, args.image_path)
    print('Creating networks and loading parameters')
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    nrof_images_total = 0
    nrof_successfully_aligned = 0

    output_class_dir = os.path.join(output_dir, cls.name)

    if not os.path.exists(output_class_dir):
        os.makedirs(output_class_dir)
    image_path = cls.image_paths
    nrof_images_total += 1
    filename = os.path.splitext(os.path.split(image_path)[1])[0]
    output_filename = os.path.join(output_class_dir, filename+'.jpeg')
    # print(image_path)
    if not os.path.exists(output_filename):
        try:
            img = imageio.imread(image_path)
        except (IOError, ValueError, IndexError) as e:
            errorMessage = '{}: {}'.format(image_path, e)
            # 在这里输出错误
            print(errorMessage)
        else:
            if img.ndim<2:
                print('Unable to align "%s"' % image_path)

            # 二维的黑白照片转换成三维的照片
            if img.ndim == 2:
                img = facenet.to_rgb(img)
            img = img[:,:,0:3]
            # left-up point + probebility
            bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

            nrof_faces = bounding_boxes.shape[0]
            if nrof_faces>0:
                det = bounding_boxes[:,0:4]
                det_arr = []
                img_size = np.asarray(img.shape)[0:2]
                if nrof_faces>1:
                    if args.detect_multiple_faces:
                        for i in range(nrof_faces):
                            det_arr.append(np.squeeze(det[i]))
                    else:
                        bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                        img_center = img_size / 2
                        offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                        offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                        index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                        det_arr.append(det[index,:])
                else:
                    det_arr.append(np.squeeze(det))

                for i, det in enumerate(det_arr):
                    det = np.squeeze(det)
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0]-args.margin/2, 0)
                    bb[1] = np.maximum(det[1]-args.margin/2, 0)
                    bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                    bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
                    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                    scaled = transform.resize(cropped, (args.image_size, args.image_size))
                    nrof_successfully_aligned += 1
                    filename_base, file_extension = os.path.splitext(output_filename)
                    if args.detect_multiple_faces:
                        output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                    else:
                        output_filename_n = "{}{}".format(filename_base, file_extension)
                    misc.imsave(output_filename_n, scaled)
            else:
                print('Unable to align "%s"' % image_path)
                sys.exit(0)

                            
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
            

# def parse_arguments(argv):
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
#     parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
#     parser.add_argument('--image_size', type=int,
#         help='Image size (height, width) in pixels.', default=182)
#     parser.add_argument('--margin', type=int,
#         help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
#     parser.add_argument('--random_order',
#         help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
#     parser.add_argument('--gpu_memory_fraction', type=float,
#         help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
#     parser.add_argument('--detect_multiple_faces', type=bool,
#                         help='Detect and align multiple faces per image.', default=False)
#     return parser.parse_args(argv)


if __name__ == '__main__':

    class argvs:
        # input_dir = '/home/hiptonese/project/face_recognition/experiment/photo/'
        output_dir = '/home/hiptonese/project/face_recognition/experiment/bbox/'
        class_name = 'dyy'
        image_size = 182
        margin = 44
        gpu_memory_fraction = 1
        detect_multiple_faces = False
        random_order = False
        image_path = '/home/hiptonese/project/face_recognition/experiment/photo/dyy/dyy_0001.jpeg'

    align_image_mtcnn(argvs)


