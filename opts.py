"""Parses command line arguments."""

import argparse
import os

import cv2
import numpy as np

import caffe
import datasets
import lib

__author__ = 'Anurag Arnab'
__copyright__ = 'Copyright (c) 2018, Anurag Arnab'
__credits__ = ['Anurag Arnab', 'Ondrej Miksik', 'Philip Torr']
__email__ = 'anurag.arnab@gmail.com'
__license__ = 'MIT'


def ParseArgs():
    """Parse command line args"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)

    # Network params
    parser.add_argument('--model_def', type=str)
    parser.add_argument('--model_weights', type=str)
    parser.add_argument('--model_name', type=str, default=None,
                        help="Used in saving results. Ignore to derive from the prototxt filename")
    parser.add_argument('--mean', type=str, default=None,
                        help='Mean should be in BGR order.')
    parser.add_argument('--mean_file', type=str)

    # IO params
    parser.add_argument('--image', type=str, help="Image to process. Ignored if image_file is present")
    parser.add_argument('--image_file', type=str, default=None)
    parser.add_argument('--is_seg', action='store_true', default=False)
    parser.add_argument('--out_dir', type=str, default='output')
    parser.add_argument('--iter_print', type=int, default=20)
    parser.add_argument('--is_verbose', action='store_true', default=False)
    parser.add_argument('--force_overwrite', action='store_true', default=False)
    parser.add_argument('--save_scores', action='store_true', default=False)
    parser.add_argument('--save_adv_example', action='store_true', default=False)
    parser.add_argument('--save_adv_image', action='store_true', default=False)
    parser.add_argument('--verbose_first', action='store_true', default=False)

    # Preprocessing
    parser.add_argument('--pad_size', type=str, help='comma separated list of <width,height>',
                        default=None)
    parser.add_argument('--resize_dims', type=str, default=None,
                        help='comma separated list of <width,height>')
    parser.add_argument('--pad_value', type=int, default=cv2.BORDER_CONSTANT,
                        help='Type of padding to do. 0 = cv2.BORDER_CONSTANT, 4 = cv2.BORDER_REFLECT_101')
    # Dilated stuff. This network uses different image pre-processing
    parser.add_argument('--is_dilated', action='store_true', default=False)

    # Adversarial attack params
    parser.add_argument('--attack_method', type=str, default='fgsm')
    parser.add_argument('--eps', type = float, default=0.25)
    parser.add_argument('--target_idx', type=int, default=0)
    parser.add_argument('--num_iters', type=int, default=-1,
                        help='Number of iterations for iterative FGSM. If left blank, formula used'
                        'by Kurakin 2016 is used.')
    parser.add_argument('--alpha', type=float, default=1,
                        help='Step size used in iterative attacks. Kurakin 2016 used 1.')
    parser.add_argument('--do_max_pert', action='store_true', default=False,
                        help='In iterative attacks, whether to stop before given number of'
                        'iterations if the max norm has been reached.')

    # Dataset params. Right now, it is only to get the palette
    parser.add_argument('--dataset', type=str, default='voc')
    parser.add_argument('--palette', type=str, default=None)
    parser.add_argument('--label_names', type=str, default=None)

    args = parser.parse_args()

    if args.mean is not None:
        args.mean = np.array([float(x) for x in args.mean.split(',')])

    if args.out_dir is not None:
        lib.CreateDir(args.out_dir)

    if args.model_name is None:
        args.model_name = os.path.basename(args.model_def).split('.')[0]

    if args.pad_size is not None:
        args.pad_size = args.pad_size.split(',')
        args.pad_size = [int(x) for x in args.pad_size]
        if len(args.pad_size) == 1:
            args.pad_size.append(args.pad_size[0])

    if args.resize_dims is not None:
        args.resize_dims = args.resize_dims.split(',')
        args.resize_dims = [int(x) for x in args.resize_dims]

    if args.dataset == 'voc':
        args.colour_map = datasets.voc.get_colour_map()
        args.label_names = datasets.voc.get_label_names()

        if args.mean is None:
            args.mean = np.array([103.939, 116.779, 123.68])
            if args.is_dilated:
                args.mean = np.array([102.93,111.36,116.52])

    elif args.dataset == 'cityscapes':
        args.label_names = datasets.cityscapes.get_label_names()
        args.colour_map = datasets.cityscapes.get_colour_map()

        if args.mean is None:
            args.mean = np.array([72.39,82.91,73.16])

    elif args.dataset == 'imagenet':
        args.label_names = open('data/synset_words.txt', 'r').readlines()
        args.label_names = [x.strip() for x in args.label_names]

    else:
        raise AssertionError('Invalid dataset')

    return args
