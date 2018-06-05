[//]: # (Image References)

[image1]: ./images/fcn8.png "FCN-8"
[image2]: ./images/deeplabv3p.png "DeepLab v3+"
[image3]: ./images/scoring.png "scoring"

# Lyft perception challenge

This is the writeup for the 7th submission from ymlai87416.

## Background

The goal of this challenge is pixel-wise identification of objects in camera
images. In other words, your task is to identify exactly what is in each pixel
of an image! More specifically, you'll be identifying cars and the drivable 
area of the road. The images below are a simulated camera image on the left 
and a label image on the right, where each different type of object in the 
image corresponds to a different color.

### Scoring criteria

![][image3]

## Implementation

### What have I done?

This is the 7th submission. In previous submission. I make use of FCN8-VGG16 [1], DeepLab v3+ [2]
and successfully obtained the following best score.

|    |      Previous score      |  Current score |
|----------|-------------:|------:|
| Final score |  79.1547 |  ?? |
| Average F score |  0.8587 | 0.865 |
| Car F score |  0.758 | 0.783 |
| Road F score |  0.9593 | 0.947 |
| FPS |  3.289 | 6.993 |


Here is one of the result I got from a previous submission

![][image1]

In this submission, I make use of the DeepLab v3+ pascal model and use transfer learning
to re-purpose it for this challenge

### DeepLab v3+

DeepLab v3+ [2] is proposed by Google and this implementation uses Xception as the backbone.
Xception [3] is also proposed by Google for predicting

The implementation and the model weighting is adopted from a Github repo bonlime/keras-deeplab-v3-plus [4].

It is written in Keras. I adopted the model, frozen the weighting in the 1st - 356th layers and trained the rest.

### Bias and Variance

This submission is for proof-of-concept only. 

#### Train, Validate and Test set

The model is trained using the 1000 images provided by Udacity at this [link](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Lyft_Challenge/Training+Data/lyft_training_data.tar.gz)

Validation set contains 172 images

Test set contains 300 images

#### Epoch, Regularization

I train the model with 10 epochs. Dropout layer follows the default implementation of 0.1

#### Transform the trained Keras model for inference optimized form

I use the script provided by Github repo: amir-abdi/keras_to_tensorflow [5] to convert my Keras model in h5 
format to a frozen tensorflow model.

I then use optimize_for_inference to further improve the network inference speed.

#### Inference rate

I cut the sky and the bottom part of the image to reduce the image size. Resizing increase inaccuracy and decrease frame 
rate so I drop it. I also do some probing on the Telsa K80 card, and find that to archive 10fps, the best input size is 
192x600, which is 115200 pixels.

The current configuration of 256x800, each frame will be processed at 0.115s. The resulting frame rate is around 7fps.



### Result

Here is a snapshot of my result. Some of the pedestrian pavement is marked as road, but the 
car is much more clear than that of my implementation of FCN8-VGG16.
 
![][image2]
 
## Reference
[1] Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

[2] Chen, Liang-Chieh, et al. "Encoder-decoder with atrous separable convolution for semantic image segmentation." arXiv preprint arXiv:1802.02611 (2018).

[3] Chollet, François. "Xception: Deep learning with depthwise separable convolutions." arXiv preprint (2016).

[4] https://github.com/bonlime/keras-deeplab-v3-plus

[5] https://github.com/amir-abdi/keras_to_tensorflow 
