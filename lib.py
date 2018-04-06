"""Useful functions for running experiments"""

import os

import cv2
import numpy as np
from PIL import Image as PILImage
import scipy.io as sio

import dilated

__author__ = 'Anurag Arnab'
__copyright__ = 'Copyright (c) 2018, Anurag Arnab'
__credits__ = ['Anurag Arnab', 'Ondrej Miksik', 'Philip Torr']
__email__ = 'anurag.arnab@gmail.com'
__license__ = 'MIT'


def CreateDir(dirname):
    """Create directory if it does not exist."""

    try:
        os.makedirs(dirname)
    except OSError, e:
        if e.errno != os.errno.EEXIST:
            raise


def SavePredictionScores(pred_scores, adv_scores, im_height, im_width, args, is_debug=False):
    """Saves the outputs of the network in a mat file."""

    pred_scores = softmax(pred_scores)
    adv_scores  = softmax(adv_scores)

    conf = pred_scores.max(axis = 0)
    adv_conf = adv_scores.max(axis = 0)

    entropy_map = entropy(pred_scores)
    conf_ratio_map = conf_ratio(pred_scores)

    adv_entropy_map = entropy(adv_scores)
    adv_conf_ratio_map = conf_ratio(adv_scores)

    model_name = args.model_name
    image_name = os.path.basename(args.image).split('.')[0]
    save_name = os.path.join(
        args.out_dir, "{}_scores_{}_eps={}.mat".format(image_name, model_name, args.eps))

    if not is_debug:
        sio.savemat(save_name, {'conf': conf, 'adv_conf': adv_conf, 'im_height' : im_height, 'im_width': im_width, 'entropy': entropy_map, 'conf_ratio': conf_ratio_map, 'adv_entropy': adv_entropy_map, 'adv_conf_ratio': adv_conf_ratio_map}, do_compression=True)
    else:
        sio.savemat(save_name, {'unary': pred_scores, 'unary_adv': adv_scores, 'conf': conf, 'adv_conf': adv_conf, 'im_height' : im_height, 'im_width': im_width, 'entropy': entropy_map, 'conf_ratio': conf_ratio_map, 'adv_entropy': adv_entropy_map, 'adv_conf_ratio': adv_conf_ratio_map}, do_compression=True)
    
    return conf


def PreprocessImage(name, pad_size, scale=None, pad_value=cv2.BORDER_REFLECT_101, 
                    resize_dims=None, args=None):
    """Preprocesses the image so that it can be fed into the network.
        name: filename of the image
        pad_size: the size which the image is padded to for input to the network.
            Images which are larger than this are resized down whilst preserving the aspect ratio,
            and the smaller side is then padded to the specified size.
        scale: Whether to resize the image by a specific scale after padding.
        pad_value: Type of padding to use, according to OpenCV.
        resize_dims: If this is passed, the image is first resized to the specified size before
            doing the padding as described above.
        args: Other command-line args passed to the program.
    """

    if args.is_dilated:
        return dilated.PreprocessImage(name, args)

    image = cv2.imread(name, 1).astype(np.float32)
    input_image = image - args.mean

    # Make image smaller, if it is bigger than pad_width x pad_height
    im_height = input_image.shape[0]
    im_width = input_image.shape[1]
    pad_width, pad_height = im_width, im_height
    if pad_size is not None:
        pad_width, pad_height = pad_size[0], pad_size[1]

    if (im_height > pad_height or im_width > pad_width) and resize_dims is None:
        if im_height > im_width:
            ratio = float( max(pad_width, pad_height) / float(im_height) )
        else:
            ratio = float( max(pad_width, pad_height) / float(im_width) )
        input_image = cv2.resize(input_image, dsize=( int(im_width * ratio), int(im_height * ratio) ) )

    elif resize_dims is not None:
        image = cv2.resize(image, dsize=(resize_dims[0], resize_dims[1]))
        input_image = cv2.resize(input_image, dsize=(resize_dims[0], resize_dims[1]))
        if pad_size is None:
            pad_height, pad_width = input_image.shape[0], input_image.shape[1]

    if scale is not None:
        if scale > 1:
            input_image = cv2.resize(input_image, dsize=(0,0), fx=scale, fy=scale,
                                     interpolation=cv2.INTER_CUBIC)
        if scale < 1:
            input_image = cv2.resize(input_image, dsize=(0,0), fx=scale, fy=scale,
                                     interpolation=cv2.INTER_AREA)

    if (pad_width is not None) and (pad_height is not None):
        input_image = cv2.copyMakeBorder(input_image, 0, pad_height - input_image.shape[0], 0,
                                         pad_width - input_image.shape[1], pad_value)

    input_image = input_image.transpose([2,0,1]) # To make it C x H x W for Caffe
    return input_image, im_height, im_width, image


def SavePredictionIm(prediction, palette, save_name):
    """Saves prediction image with a colour map"""

    prediction = prediction.astype(np.uint8)
    prediction_im = PILImage.fromarray(prediction)
    prediction_im.putpalette(palette)
    prediction_im.save(save_name)


def softmax(x, axis=0):

    y = np.exp(x)
    denominator = np.sum(y,axis = axis)
    return y / denominator


def entropy(x, axis=0):

    epsilon = np.finfo(float).eps
    y = -np.sum (x * np.log(x + epsilon), axis = 0)
    return y


def conf_ratio(x, axis=0):
    """Ratio of most-confident to second-most confident prediction.
       Done in a quick and dirty way: the scores are sorted first.
    """

    x_sorted = np.sort(x, axis=axis)
    channels = x_sorted.shape[axis]

    if channels < 2:
        raise AssertionError("Need at least two channels")

    epsilon = np.finfo(float).eps
    return x_sorted[channels-1,:,:] / (x_sorted[channels-2,:,:] + epsilon)
