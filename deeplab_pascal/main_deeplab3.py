import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import sys
import numpy as np
import time

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

import matplotlib as mpl
import matplotlib.pyplot as plt

gg = []

def crop_image(image):
    return image[274:530,:,:]

def pad_image(image):
    result = np.zeros((600,800))
    result[274:530,:] = image
    return result

from sklearn.metrics import f1_score, accuracy_score

def mean_iou_score(y_pred, y_true, label_set):
    class_iou_list = []
    for i in label_set:
        intersect = np.sum(np.logical_and((y_pred == i), (y_true == i)))
        union = np.sum(np.logical_or((y_pred == i), (y_true == i)))
        if union != 0:
            class_iou_list.append(intersect * 1. / union)

    return np.mean(class_iou_list)


def gather_training_stats(sess, epoch, get_batches_fn, logits, keep_prob, input_image, writer):
    # run all the batch again
    batch_size = 32
    mean_iou_list = []
    accuracy_list = []
    fscore_list = []
    label_set = [i for i in range(13)]

    for image, label in get_batches_fn(batch_size):
        curr_batch_size = len(label)
        image_shape = (len(label[0]), len(label[0][0]))
        softmax = sess.run([tf.nn.softmax(logits)],
                           {keep_prob: 1.0, input_image: image})
        softmax = np.argmax(np.array(softmax), axis=2)
        softmax = np.reshape(softmax, (curr_batch_size, image_shape[0] * image_shape[1]))
        label = np.argmax(np.array(label), axis=3)
        label = np.reshape(label, (curr_batch_size, image_shape[0] * image_shape[1]))

        for i in range(curr_batch_size):
            mean_iou_list.append(mean_iou_score(softmax[i], label[i], label_set))
            accuracy_list.append(accuracy_score(softmax[i], label[i]))
            fscore_list.append(f1_score(softmax[i], label[i], average='macro'))

    mean_iou_r = np.mean(mean_iou_list)
    accuracy_r = np.mean(accuracy_list)
    fscore_r = np.mean(fscore_list)

    summary = tf.Summary()
    summary.value.add(tag="training_accuracy", simple_value=accuracy_r)
    summary.value.add(tag="training_iou", simple_value=mean_iou_r)
    summary.value.add(tag="training_fscore", simple_value=fscore_r)
    writer.add_summary(summary, epoch)


def run():
    num_classes = 13
    image_shape = (608, 800)
    data_dir = '../data'
    runs_dir = './runs'
    epochs = 10
    batch_size = 16

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "train"

    print("Current mode: ", mode)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    LOGDIR = os.path.join('./data', 'fcn8_log')

    if mode == "train":

        pass

    elif mode == "test":
        if len(sys.argv) < 3:
            print("main.py test <graph location>")
        else:
            graph_file = sys.argv[2]
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

                t0 = time.time()
                # Save inference data using helper.save_inference_samples
                output_dir = helper.save_inference_samples_3(runs_dir, data_dir, sess, image_shape, out, None, x)
                duration = time.time() - t0
                print("Run complete, time taken = {0}".format(duration))

                helper.calculate_score(os.path.join(data_dir, 'Validate', 'CameraSeg'), output_dir)
    else:
        print("Command unrecognized.")


if __name__ == '__main__':
    run()
