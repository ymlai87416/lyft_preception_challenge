import sys, skvideo.io, json, base64
from PIL import Image
from io import BytesIO
import tensorflow as tf
import sys
import numpy as np
import scipy
import os
import cv2
from tqdm import tqdm

file = sys.argv[-1]

if file == 'demo.py':
    print("Error loading video")
    quit


def get_model_path(model_name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name)


# Define encoder function
def encode(array):
    retval, buffer = cv2.imencode('.png', array)
    return base64.b64encode(buffer).decode("utf-8")


def crop_image(image):
    return image[274:530, :, :]


def pad_image(image):
    result = np.zeros((600, 800))
    result[274:530, :] = image
    return result


video = skvideo.io.vread(file)

answer_key = {}

# Frame numbering starts at 1
frame = 1

graph_file = get_model_path('deeplab_10_trim_xception.opt.h5.pb')
image_shape = (256, 800)

use_xla = False

config = tf.ConfigProto()
if use_xla:
    jit_level = tf.OptimizerOptions.ON_1
    config.graph_options.optimizer_options.global_jit_level = jit_level

with tf.Session(graph=tf.Graph(), config=config) as sess:
    gd = tf.GraphDef()
    g = sess.graph
    with tf.gfile.Open(graph_file, 'rb') as f:
        data = f.read()
        gd.ParseFromString(data)
    tf.import_graph_def(gd, name='')
    x = g.get_tensor_by_name('input_1:0')
    out = g.get_tensor_by_name('output_node0:0')

    for rgb_frame in video:
        # image_origin_shape = (rgb_frame.shape[0], rgb_frame.shape[1])
        # image = scipy.misc.imresize(rgb_frame, image_shape)
        # image = cv2.resize(rgb_frame, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_CUBIC)
        image = crop_image(rgb_frame)

        image = image / 127.5 - 1.
        im_softmax = sess.run(
            out,
            {x: [image]})
        num_classes = im_softmax.shape[3]
        im_softmax = im_softmax.reshape(image_shape[0], image_shape[1], num_classes)

        im_argmax = np.argmax(im_softmax, axis=2).reshape(image_shape[0], image_shape[1])
        # im_argmax = scipy.misc.imresize(im_argmax.astype(np.uint8), image_origin_shape, interp='nearest')
        # im_argmax = cv2.resize(im_argmax.astype(np.uint8), (image_origin_shape[1], image_origin_shape[0]), interpolation=cv2.INTER_NEAREST)
        im_argmax = pad_image(im_argmax)

        # Look for red cars :)
        binary_car_result = np.where(im_argmax == 10, 1, 0).astype('uint8')

        # Look for road :)
        binary_road_result = np.where(im_argmax == 7, 1, 0).astype('uint8')

        answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]

        # Increment frame
        frame += 1

# Print output in proper json format
print(json.dumps(answer_key))
