import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import cv2
import time
import sklearn.metrics

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def apply_random_shadow(image):
    #
    # Add a random shadow to a BGR image to pretend
    # we've got clouds or other interference on the road.
    #
    rows, cols, _ = image.shape
    top_y = cols * np.random.uniform()
    top_x = 0
    bot_x = rows
    bot_y = cols * np.random.uniform()
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = 0 * image_hls[:, :, 1]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1

    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = (shadow_mask == 1)
        cond0 = (shadow_mask == 0)
        if np.random.randint(2) == 1:
            image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
        else:
            image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright

    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
    return image


def apply_brightness_augmentation(image):
    #
    # expects input image as BGR, adjusts brightness to
    # pretend we're in different lighting conditions.
    #
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image2 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image2


def apply_translation(image, label, translation_range):
    #
    # Shift image up or down a bit within trans_range pixels,
    # filling missing area with black.  IMG is in BGR format.
    #
    rows, cols, _ = image.shape
    tr_x = translation_range * np.random.uniform() - translation_range / 2
    tr_y = 10 * np.random.uniform() - 10 / 2
    trans_m = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    img_tr = cv2.warpAffine(image, trans_m, (cols, rows))
    label_tr = cv2.warpAffine(label, trans_m, (cols, rows))
    return img_tr, label_tr


def augment(image_raw, label):
    img = apply_brightness_augmentation(image_raw)

    if np.random.randint(4) == 0:
        img_shadows = apply_random_shadow(img)
    else:
        img_shadows = img

    if np.random.randint(2) == 0:
        img_trans, label_trans = apply_translation(img_shadows, label, 25)
    else:
        img_trans, label_trans = img_shadows, label

    if np.random.randint(4) == 0:
        img_flip = cv2.flip(img_trans, 1)
        label_flip = cv2.flip(label_trans, 1)
    else:
        img_flip, label_flip = img_trans, label_trans

    return img_flip, label_flip

def preprocess_labels(label_image):
    # Identify lane marking pixels (label is 6)
    labels_new = np.copy(label_image)
    lane_marking_pixels = (label_image[:,:,0] == 6).nonzero()
    # Set lane marking pixels to road (label is 7)
    labels_new[lane_marking_pixels] = 7

    # Identify all vehicle pixels
    vehicle_pixels = (label_image[:,:,0] == 10).nonzero()
    # Isolate vehicle pixels associated with the hood (y-position > 496)
    hood_indices = (vehicle_pixels[0] >= 496).nonzero()[0]
    hood_pixels = (vehicle_pixels[0][hood_indices], \
                   vehicle_pixels[1][hood_indices])
    # Set hood pixel labels to 0
    labels_new[hood_pixels] = 0
    # Return the preprocessed label image
    return labels_new


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """

    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'CameraRGB', '*.png'))
        label_paths = {
            os.path.basename(path): path
            for path in glob(os.path.join(data_folder, 'CameraSeg', '*.png'))}

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i + batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image_p = scipy.misc.imresize(preprocess_labels(scipy.misc.imread(gt_image_file)), image_shape)

                image_a, gt_image_a = augment(image, gt_image_p)

                gt_image_a_tmp = gt_image_a[:, :, 0:1]

                gt_image_a = None
                for i in range(13):
                    gt_temp = np.all(gt_image_a_tmp == i, axis=2)
                    gt_temp = gt_temp.reshape(*gt_temp.shape, 1)
                    if gt_image_a is None:
                        gt_image_a = gt_temp
                    else:
                        gt_image_a = np.concatenate((gt_image_a, gt_temp), axis=2)

                images.append(image_a)
                gt_images.append(gt_image_a)

            yield np.array(images), np.array(gt_images)

    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'CameraRGB', '*.png')):
        image_origin = scipy.misc.imread(image_file)
        image_origin_shape = (image_origin.shape[0], image_origin.shape[1])
        image = scipy.misc.imresize(image_origin, image_shape)

        t0 = time.time()
        if keep_prob is None:
            im_softmax = sess.run(
                [tf.nn.softmax(logits)],
                {image_pl: [image]})
        else:
            im_softmax = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 1.0, image_pl: [image]})

        duration = time.time() - t0
        print("Run {0} complete, time taken = {1}".format(image_file, duration))
        #num_classes = im_softmax[0].shape[1]
        num_classes = im_softmax[0].shape[3]
        im_softmax = im_softmax[0].reshape(image_shape[0], image_shape[1], num_classes)
        im_argmax = np.argsort(im_softmax, axis=2)[:, :, num_classes - 1].reshape(image_shape[0], image_shape[1])
        im_argmax = scipy.misc.imresize(im_argmax.astype(np.uint8), image_origin_shape, interp='nearest').reshape(image_origin_shape[0], image_origin_shape[1], 1)

        im_blue = np.zeros((image_origin_shape[0], image_origin_shape[1], 1), dtype=np.uint8)
        im_green = np.zeros((image_origin_shape[0], image_origin_shape[1], 1), dtype=np.uint8)

        final_im = np.concatenate((im_argmax, im_blue, im_green), axis=2)

        yield os.path.basename(image_file), np.array(final_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'Test'), image_shape)
    for name, image in image_outputs:
        scipy.misc.toimage(image, cmin=0, cmax=255).save(os.path.join(output_dir, name))

    return output_dir

def gen_test_output_2(sess, softmax, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'CameraRGB', '*.png')):
        image_origin = scipy.misc.imread(image_file)
        image_origin_shape = (image_origin.shape[0], image_origin.shape[1])
        image = scipy.misc.imresize(image_origin, image_shape)

        if keep_prob is None: 
            im_softmax = sess.run(
                softmax,
                {image_pl: [image]})
        else:
            im_softmax = sess.run(
                softmax,
                {keep_prob: 1.0, image_pl: [image]})
        
        num_classes = im_softmax.shape[1]
        im_softmax = im_softmax.reshape(image_shape[0], image_shape[1], num_classes)

        im_argmax = np.argsort(im_softmax, axis=2)[:, :, num_classes - 1].reshape(image_shape[0], image_shape[1])
        im_argmax = scipy.misc.imresize(im_argmax.astype(np.uint8), image_origin_shape, interp='nearest').reshape(image_origin_shape[0], image_origin_shape[1], 1)

        im_blue = np.zeros((image_origin_shape[0], image_origin_shape[1], 1), dtype=np.uint8)
        im_green = np.zeros((image_origin_shape[0], image_origin_shape[1], 1), dtype=np.uint8)

        final_im = np.concatenate((im_argmax, im_blue, im_green), axis=2)

        yield os.path.basename(image_file), np.array(final_im)


def save_inference_samples_2(runs_dir, data_dir, sess, image_shape, softmax, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output_2(
        sess, softmax, keep_prob, input_image, os.path.join(data_dir, 'Test'), image_shape)
    for name, image in image_outputs:
        scipy.misc.toimage(image, cmin=0, cmax=255).save(os.path.join(output_dir, name))

    return output_dir

def calculate_score(true_dir, predict_dir):
    vehicle_score_list = []
    road_score_list = []
    score_list = []

    for label_file in glob(os.path.join(true_dir, '*.png')):
        base_file_name = os.path.basename(label_file)
        true_label_fn = os.path.join(true_dir, base_file_name)
        pred_label_fn = os.path.join(predict_dir, base_file_name)

        sample_label = scipy.misc.imread(true_label_fn)
        y_true = preprocess_labels(sample_label)[:, :, 0].ravel()
        y_pred = scipy.misc.imread(pred_label_fn)[:, :, 0].ravel()

        y_road_true = (y_true == 7)
        y_road_pred = (y_pred == 7)
        y_vehicle_true = (y_true == 10)
        y_vehicle_pred = (y_pred == 10)

        vehicle_score = sklearn.metrics.fbeta_score(y_vehicle_true, y_vehicle_pred, 2)
        road_score = sklearn.metrics.fbeta_score(y_road_true, y_road_pred, 0.5)
        score = (vehicle_score + road_score) / 2

        vehicle_score_list.append(vehicle_score)
        road_score_list.append(road_score)
        score_list.append(score)

    print("Vehicle score = %.3f, std=%.3f" % (np.mean(vehicle_score_list), np.std(vehicle_score_list)))
    print("Road score = %.3f, std=%.3f" % (np.mean(road_score_list), np.std(road_score_list)))
    print("Submission score = %.3f, std=%.3f" % (np.mean(score_list), np.std(score_list)))

