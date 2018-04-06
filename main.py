"""Generates adversarial examples for segmentation and image classification models."""

import sys
import os
import datetime
import shutil

import cv2
import numpy as np
import scipy.io as sio
import caffe

import attacks
import dilated
import lib
import opts

__author__ = 'Anurag Arnab'
__copyright__ = 'Copyright (c) 2018, Anurag Arnab'
__credits__ = ['Anurag Arnab', 'Ondrej Miksik', 'Philip Torr']
__email__ = 'anurag.arnab@gmail.com'
__license__ = 'MIT'


adv_attacks = {
    'fgsm': attacks.fgsm,
    'targetted_fgsm': attacks.fgsm_targetted,
    'iterative_fgsm': attacks.IterativeFGSM,
    'iterative_fgsm_ll': attacks.IterativeFGSMLeastLikely
}


def Predict(net, x, dummy_label=None, label_names=None, do_top_5=False, is_seg=False):
    """Performs a forward pass of the image.

       net: The caffe network. Assumes that the network definition has the following keys:
        "data" - input image
        "label" - the label used to compute the loss.
        "output" - predicted logits by the network.
       x: The data to be passed through the network
       dummy_label: Model definitions for adversarial examples have a label input as well
                    for computing the loss. For only prediction, this can be set arbitrarily
       label_names: The names of each class
       do_top_5: For image classification models, whether to show top 5 predictions
       is_seg: Whether 'net' is a segmentation model (true) or image classification model (false)
    """

    net.blobs['data'].data[0,:,:,:] = np.squeeze(x)

    if dummy_label is None:
        net.blobs['label'].data[...] = np.zeros( net.blobs['label'].data.shape )
    else:
        net.blobs['label'].data[...] = dummy_label

    net.forward()

    net_prediction = net.blobs['output'].data[0].argmax(axis=0).astype(np.uint32)
    confidence = net.blobs['output'].data[0].astype(np.float32)

    if is_seg:
        return net_prediction, confidence

    if label_names is not None:
        pred_label = label_names[net_prediction]
    else:
        pred_label = 'Unknown corpus'

    if do_top_5:
        scores = net.blobs['output'].data[0]
        indices = np.argsort( -scores )
        indices = indices[0:5]
        print indices
        for index in indices:
            print "{:60} {:0.3f}".format(label_names[index],scores[index])
        print ""

    return net_prediction, confidence, pred_label


def GetAdvFuncArgs(args, net, x):
    """Prepares keyword arguments for adversarial attack functions."""

    attack_method = args.attack_method.lower()
    adv_args = {}

    if attack_method == 'fgsm':
        adv_args = {
            'net': net,
            'x': x,
            'eps': args.eps,
        }
    elif attack_method == 'targetted_fgsm':
        adv_args = {
            'net': net,
            'x': x,
            'eps': args.eps,
            'target_idx': args.target_idx
        }
    elif attack_method in ['iterative_fgsm', 'iterative_fgsm_ll']:
        adv_args = {
            'net': net,
            'x': x,
            'eps': args.eps,
            'num_iters': args.num_iters,
            'alpha': args.alpha,
            'do_stop_max_pert': args.do_max_pert
        }
    else:
        raise AssertionError('Unknown attack method')

    return adv_args


def PredictWrapper(net, image, orig_image, dummy_label=None, is_seg=True, args=None):
    """Wrapper for calling the Predict function. DilatedNet has its own pre- and post-processing."""

    if not args.is_dilated:
        pred, conf = Predict(net, image, dummy_label=dummy_label, is_seg=is_seg)
    else:
        _, conf = Predict(net, image, dummy_label=dummy_label, is_seg=is_seg)
        pred = dilated.PostprocessPrediction(conf, orig_image, args.dataset)
        pred = pred.argmax(axis=0).astype(np.uint32)

    pred = pred[0:orig_image.shape[0], 0:orig_image.shape[1]]
    conf = conf[:, 0:orig_image.shape[0], 0:orig_image.shape[1]]
    return pred, conf


def CheckAlreadyProcessed(args, image_name, model_name):
    """Checks if the image has already been saved in the output directory."""

    if args.force_overwrite:
        return False

    if args.save_adv_example:
        output_template = "{}_advinput_{}_eps={}_target_idx={}.mat"
    else:
        output_template = "{}_pert_{}_eps={}_target_idx={}.png"
    output_name = output_template.format(
        image_name, model_name, args.eps, args.target_idx)

    if os.path.exists( os.path.join(args.out_dir, output_name)) :
        return True

    return False


def CheckAllDone(args):
    """Checks if all images have already been saved in the output directory."""

    image_names = open(args.image_file, 'r').readlines()
    image_names = [x.strip() for x in image_names]

    for im_path in image_names:
        im_name = os.path.basename(im_path).split('.')[0].replace('_leftImg8bit', '')

        is_done = CheckAlreadyProcessed(args, im_name, args.model_name)
        if not is_done:
            return False

    return True


def main_seg(args, net=None, is_debug=False):
    """Adversarial example for segemtantion models"""

    colour_map = args.colour_map
    model_name = args.model_name
    image_name = os.path.basename(args.image).split('.')[0].replace('_leftImg8bit', '')

    if CheckAlreadyProcessed(args, image_name, model_name):
        return

    if (args.gpu >= 0 and net is None):
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu)

    if net is None:
        net = caffe.Net(args.model_def, args.model_weights, caffe.TEST)

    image, im_height, im_width, orig_image = lib.PreprocessImage(
        args.image, args.pad_size, pad_value=args.pad_value, resize_dims=args.resize_dims, args=args)

    orig_pred, orig_conf = PredictWrapper(
        net, image, orig_image, dummy_label=None, is_seg=True, args=args)
    save_name = os.path.join(args.out_dir,
                             "{}_original_pred_{}.png".format(image_name, model_name))
    lib.SavePredictionIm(orig_pred, colour_map, save_name)

    adv_func_args = GetAdvFuncArgs(args, net, image)
    adversarial_image_data, added_noise_data = adv_attacks[args.attack_method](**adv_func_args)

    adv_pred, adv_conf = PredictWrapper(
        net, adversarial_image_data, orig_image, dummy_label=None, is_seg=True, args=args)
    save_name =  os.path.join(args.out_dir,
                              "{}_adversarial_pred_{}_eps={}_target_idx={}.png".format(
                              image_name, model_name, args.eps, args.target_idx))
    lib.SavePredictionIm(adv_pred, colour_map, save_name)

    if args.save_adv_image:
        adversarial_image = np.transpose( np.squeeze(adversarial_image_data[0,:,:,:]), [1,2,0] )
        save_name = os.path.join(args.out_dir,
                                 "{}_adversarial_example_{}_eps={}_target_idx={}.png".format(
                                 image_name, model_name, args.eps, args.target_idx) )
        cv2.imwrite(save_name, adversarial_image + args.mean)

    added_pert = np.transpose( np.squeeze(added_noise_data[0,:,:,:]), [1,2,0] )
    save_name = os.path.join(args.out_dir,
                             "{}_pert_{}_eps={}_target_idx={}.png".format(
                             image_name, model_name,   args.eps, args.target_idx) )
    cv2.imwrite(save_name, added_pert + args.mean)

    if args.save_scores:
        lib.SavePredictionScores(orig_conf, adv_conf, im_height, im_width, args, is_debug)

    if args.save_adv_example:
        output_template = "{}_advinput_{}_eps={}_target_idx={}.mat"
        output_name = output_template.format(
            image_name, model_name, args.eps, args.target_idx)
        output_name = os.path.join(args.out_dir, output_name)
        sio.savemat(output_name, {'advinput': added_noise_data.astype(np.float32)})


def main_image(args):
    """Adversarial example for ImageNet classification model."""

    if args.gpu >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu)

    net = caffe.Net(args.model_def, args.model_weights, caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', args.mean)
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)

    image = caffe.io.load_image(args.image)
    image = transformer.preprocess('data', image)

    print "Prediction of original image"
    Predict(net, image, do_top_5=True, label_names=args.label_names)

    adv_func_args = GetAdvFuncArgs(args, net, image)
    adversarial_image_data, added_noise_data = adv_attacks[args.attack_method](**adv_func_args)

    print "Prediction of adversarial image"
    Predict(net, adversarial_image_data, do_top_5=True, label_names=args.label_names)

    adversarial_image = np.squeeze(adversarial_image_data[0,:,:,:]) # CxHxW
    adversarial_image = np.transpose( adversarial_image, [1,2,0] ) # HxWxC
    cv2.imwrite("adversarial_example.png", adversarial_image + args.mean)

    added_noise = np.transpose( np.squeeze(added_noise_data[0,:,:,:]), [1,2,0] )
    cv2.imwrite("perturbation.png", added_noise + args.mean)


def main_batch(args):
    """Adversarial examples for semantic segmentation models evaluated over all images in a list file."""

    if CheckAllDone(args):
        print "Processing", args.image_file
        print "Arguements", sys.argv
        print "Entire experiment is already done. Quitting"
        return

    if args.gpu >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu)

    model_name = os.path.basename(args.model_def).split('.')[0]
    lib.CreateDir(args.out_dir)

    ## Save the command line args used to run the program
    f_cmdline = open( os.path.join(
        args.out_dir, 'cmdline' + str(datetime.datetime.now()).replace(' ','_') + '.txt'), 'w')
    for i in range(0, len(sys.argv)):
        f_cmdline.write(sys.argv[i] + " ")
    f_cmdline.close()

    ## Copy the model prototxt to the folder
    shutil.copyfile(args.model_def, os.path.join(args.out_dir, model_name + '.prototxt' ) )

    ## Create the network and start the actual experiment
    net = caffe.Net(args.model_def, args.model_weights, caffe.TEST)
    print "Running batch images on", args.image_file
    print "Arguments", sys.argv

    image_names = open(args.image_file, 'r').readlines()
    image_names = [x.strip() for x in image_names]

    for i, im_name in enumerate(image_names):
        args.image = im_name
        main_seg(args, net)

        if i % args.iter_print == 0:
            time_str = str(datetime.datetime.now())
            print "[{}] Image {}: {}".format(time_str, i, im_name)
            sys.stdout.flush()


if __name__ == '__main__':
    args = opts.ParseArgs()

    if args.image_file is not None:
        main_batch(args)
    elif args.is_seg:
        main_seg(args)
    else:
        main_image(args)
