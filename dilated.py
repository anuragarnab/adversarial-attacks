"""Pre-processing and post-processing steps for the Dilated Convolutions network,
   according to original author's implementation.
   https://github.com/fyu/dilation/blob/master/predict.py"""

import cv2
import numpy as np
import numba

__author__ = 'Anurag Arnab'
__copyright__ = 'Copyright (c) 2018, Anurag Arnab'
__credits__ = ['Anurag Arnab', 'Ondrej Miksik', 'Philip Torr']
__email__ = 'anurag.arnab@gmail.com'
__license__ = 'MIT'


@numba.jit(nopython=True)
def interp_map(prob, zoom, width, height):
    zoom_prob = np.zeros((prob.shape[0], height, width), dtype=np.float32)
    for c in range(prob.shape[0]):
        for h in range(height):
            for w in range(width):
                r0 = h // zoom
                r1 = r0 + 1
                c0 = w // zoom
                c1 = c0 + 1
                rt = float(h) / zoom - r0
                ct = float(w) / zoom - c0
                v0 = rt * prob[c, r1, c0] + (1 - rt) * prob[c, r0, c0]
                v1 = rt * prob[c, r1, c1] + (1 - rt) * prob[c, r0, c1]
                zoom_prob[c, h, w] = (1 - ct) * v0 + ct * v1
    return zoom_prob


def PreprocessImage(name, args):
    """Preprocess according to the original author's code."""

    image = cv2.imread(name, 1).astype(np.float32) - args.mean

    if args.resize_dims is not None:
        image = cv2.resize(image, dsize = (args.resize_dims[0], args.resize_dims[1]))

    im_height = image.shape[0]
    im_width = image.shape[1]

    label_margin = 186
    input_image = cv2.copyMakeBorder(image, label_margin, label_margin,
                                     label_margin, label_margin, cv2.BORDER_REFLECT_101)
    input_size = [args.pad_size[1], args.pad_size[0]] # Order is H x W
    margin = [0, input_size[0] - input_image.shape[0],
              0, input_size[1] - input_image.shape[1]]

    input_image = cv2.copyMakeBorder(input_image, margin[0], margin[1], margin[2],
                                     margin[3], cv2.BORDER_REFLECT_101)

    input_image = input_image.transpose([2,0,1]) # To make it C x H x W
    return input_image, im_height, im_width, image


def PostprocessPrediction(x, image, dataset, zoom=8):
    """Postprocess according to the original author's code."""

    if dataset.lower() == 'cityscapes':
        return x[:,0:image.shape[0],0:image.shape[1]] # Caffe blob is CxHxW
    elif dataset.lower() == 'voc':
        return interp_map(x, zoom=zoom, width =image.shape[1], height =image.shape[0])
    else:
        raise AssertionError('Unknown dataset. Dataset is '+ dataset + '\n')
