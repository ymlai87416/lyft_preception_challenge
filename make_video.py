import os
from glob import glob
import scipy
import numpy as np
from PIL import Image
import scipy.misc

def make_video(outvid, images, fps=5, size=None,
               is_color=True, format="XVID"):
    """
    Create a video from a list of images.
 
    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
 
    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid

def save(video_img_dir, video_fname):
    command = "ffmpeg -r 1 -i {0}/img%01d.png -vcodec mpeg4 -y {1}" % (video_img_dir, video_fname)
    os.system(command)


def preprocess(input_img, input_label, video_img_dir):
    for image_fn in glob(os.path.join(input_img, '*.png')):
        base_file_name = os.path.basename(image_fn)
        pred_label_fn = os.path.join(input_label, base_file_name)

        image = scipy.misc.imread(image_fn)
        label = scipy.misc.imread(pred_label_fn)[:, :, 0:1]

        binary_car_result = label == 10

        binary_road_result = label == 7

        # Apply road judgement to original image as a mask with alpha = 50%
        mask_road = np.dot(binary_road_result, np.array([[0, 255, 0, 127]]))
        mask_car = np.dot(binary_car_result, np.array([[0, 0, 255, 127]]))

        mask_road = scipy.misc.toimage(mask_road, mode="RGBA")
        mask_car = scipy.misc.toimage(mask_car, mode="RGBA")

        street_img = Image.fromarray(image)
        street_img.paste(mask_road, box=None, mask=mask_road)
        street_img.paste(mask_car, box=None, mask=mask_car)

        street_img.save(os.path.join(video_img_dir, base_file_name))


input_img = './data/Test/CameraRGB'
input_label = './deeplab_pascal/runs/1528159274.8549316'
video_img_dir = './video'
video_fname = 'example.mp4'

preprocess(input_img, input_label, video_img_dir)


save(video_img_dir, video_fname)