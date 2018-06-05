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


def debug_show_image(image):
    plt.ion()
    plt.figure()
    plt.imshow(image)
    plt.ioff()
    plt.show()


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(
        sess,
        [vgg_tag],
        vgg_path)

    detection_graph = tf.get_default_graph()
    vgg_input_tensor = detection_graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor = detection_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out_tensor = detection_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out_tensor = detection_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out_tensor = detection_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    fcn_layer7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='SAME',
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                           name='fcn_layer7_conv_1x1')

    fcn_layer7_deconv = tf.layers.conv2d_transpose(fcn_layer7_conv_1x1, num_classes, 4, 2, padding='SAME',
                                                   kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                                   name='fcn_layer7_deconv')

    vgg_layer4_out_scale = tf.multiply(vgg_layer4_out, 0.01, name='vgg_layer4_out_scale')

    fcn_layer4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out_scale, num_classes, 1, padding='SAME',
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                           name='fcn_layer4_conv_1x1')

    intermediate_1 = tf.add(fcn_layer7_deconv, fcn_layer4_conv_1x1, name='intermediate_1')

    vgg_layer3_out_scale = tf.multiply(vgg_layer3_out, 0.0001, name='vgg_layer3_out_scale')

    fcn_layer3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out_scale, num_classes, 1, padding='SAME',
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                           name='fcn_layer3_conv_1x1')

    intermediate_1_deconv = tf.layers.conv2d_transpose(intermediate_1, num_classes, 4, 2, padding='SAME',
                                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                                       name='intermediate_1_deconv')

    intermediate_2 = tf.add(intermediate_1_deconv, fcn_layer3_conv_1x1, name='intermediate_2')

    fcn_output = tf.layers.conv2d_transpose(intermediate_2, num_classes, 16, 8, padding='SAME',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                            name='fcn_output')

    return fcn_output


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # freeze all convolution variables
    tvars = tf.trainable_variables()
    trainable_vars = [var for var in tvars if not (var.name.startswith('conv'))]

    # print("Trainable parameters are: ")
    # for var in trainable_vars:
    #    print(var.name + "\n")

    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="logit")
    tf.nn.softmax(logits, name="Softmax")  # output layer
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
    # add regularization to the loss
    reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    tf.summary.scalar('regularization_loss', reg_losses)
    reg_constant = 0.01
    loss = cross_entropy_loss + reg_constant * reg_losses

    tf.summary.scalar('total_loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_operation = optimizer.minimize(cross_entropy_loss, var_list=trainable_vars)

    return logits, training_operation, loss


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, writer=None, merged=None, logits=None):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    :param writer: Tensorboard writer
    :param merged: Tensorboard merged summary
    """
    sess.run(tf.global_variables_initializer())

    keep_prob_value = 0.5
    learning_rate_value = 0.001

    show_image = False

    step = 0
    for i in range(epochs):
        print("Epoch: = {:d}".format(i))
        for image, label in get_batches_fn(batch_size):
            # try:
            #    if not show_image:
            #        print(image[0].shape)
            #        #debug_show_image(image[0])
            #        plt.imsave("./data/training_sample.png", image[0])
            #        show_image = True
            # except:
            #    print("Oops! error")

            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: label, keep_prob: keep_prob_value,
                                          learning_rate: learning_rate_value})

            if step % 5 == 0 and writer is not None and merged is not None:
                summary = sess.run(merged, feed_dict={input_image: image, correct_label: label, keep_prob: 1,
                                                      learning_rate: learning_rate_value})
                writer.add_summary(summary, step)

            step = step + 1

        if writer is not None and logits is not None:
            gather_training_stats(sess, i, get_batches_fn, logits, keep_prob, input_image, writer)

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

        with tf.Session() as sess:
            # Path to vgg model
            vgg_path = os.path.join(data_dir, 'vgg')
            # Create function to get batches
            get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'Train'), image_shape)

            # OPTIONAL: Augment Images for better results
            #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

            # Build NN using load_vgg, layers, and optimize function
            input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
            layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

            correct_label = tf.placeholder(tf.int32, (None, image_shape[0], image_shape[1], num_classes),
                                           name='correct_label')
            learning_rate = tf.placeholder(tf.float32, name='learning_rate')

            logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(LOGDIR, graph=sess.graph)

            # Train NN using the train_nn function
            train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                     correct_label, keep_prob, learning_rate, train_writer, merged, logits=logits)

            # Save the model for future use
            # as_text = true will cause freeze_graph throw memory error
            tf.train.write_graph(sess.graph_def, './fcn8', 'base_graph.pb', as_text=False)
            print("Model graph saved in path: ./fcn8/base_graph.pb")
            saver = tf.train.Saver()
            save_path = saver.save(sess, "./fcn8/ckpt")
            print("Model weights saved in path: %s" % save_path)

            t0 = time.time()
            # Save inference data using helper.save_inference_samples
            output_dir = helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
            duration = time.time() - t0
            print("Run complete, time taken = {0}".format(duration))

            helper.calculate_score(os.path.join(data_dir, 'Test', 'CameraSeg'), output_dir)

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
                x = g.get_tensor_by_name('input_2:0')
                out = g.get_tensor_by_name('output_node0:0')

                t0 = time.time()
                # Save inference data using helper.save_inference_samples
                output_dir = helper.save_inference_samples_3(runs_dir, data_dir, sess, image_shape, out, None, x)
                duration = time.time() - t0
                print("Run complete, time taken = {0}".format(duration))

                helper.calculate_score(os.path.join(data_dir, 'Test', 'CameraSeg'), output_dir)
    else:
        print("Command unrecognized.")


if __name__ == '__main__':
    run()
