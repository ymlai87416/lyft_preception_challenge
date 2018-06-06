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

file = sys.argv[1]
outfile = sys.argv[2]


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

writer = skvideo.io.FFmpegWriter(outfile, inputdict={'-r': '10'}, outputdict={'-vcodec': 'mpeg4', '-b': '2000k'})

answer_key = {}

# Frame numbering starts at 1
frame = 1

graph_file = './submissions/final/deeplab_10_trim_xception_a.opt.h5.pb'
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
        image = crop_image(rgb_frame)

        image = image / 127.5 - 1.
        im_softmax = sess.run(
            out,
            {x: [image]})
        num_classes = im_softmax.shape[3]
        im_softmax = im_softmax.reshape(image_shape[0], image_shape[1], num_classes)

        im_argmax = np.argmax(im_softmax, axis=2).reshape(image_shape[0], image_shape[1])
        im_argmax = pad_image(im_argmax)

        # Look for red cars :)
        binary_car_result = np.where(im_argmax == 10, 1, 0).astype('uint8').reshape(600, 800, 1)

        # Look for road :)
        binary_road_result = np.where(im_argmax == 7, 1, 0).astype('uint8').reshape(600, 800, 1)

        mask_road = np.dot(binary_road_result, np.array([[0, 255, 0, 127]]))
        mask_car = np.dot(binary_car_result, np.array([[0, 0, 255, 127]]))

        mask_road = scipy.misc.toimage(mask_road, mode="RGBA")
        mask_car = scipy.misc.toimage(mask_car, mode="RGBA")

        street_img = Image.fromarray(rgb_frame)
        street_img.paste(mask_road, box=None, mask=mask_road)
        street_img.paste(mask_car, box=None, mask=mask_car)
        writer.writeFrame(np.array(street_img))

        # Increment frame
        frame += 1

writer.close()
