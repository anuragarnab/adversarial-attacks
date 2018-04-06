"""Generates adversarial example for Caffe networks."""

import numpy as np
import caffe

__author__ = 'Anurag Arnab'
__copyright__ = 'Copyright (c) 2018, Anurag Arnab'
__credits__ = ['Anurag Arnab', 'Ondrej Miksik', 'Philip Torr']
__email__ = 'anurag.arnab@gmail.com'
__license__ = 'MIT'


def fgsm(net, x, eps):
    r"""Caffe implementation of the Fast Gradient Sign Method.
    This attack was proposed in
    net: The Caffe network. Must have its weights initialised already
         Makes the following assumptions
            - force_backward is set to "true" so that gradients are computed
            - Has two inputs: "data" and "label"
            - Has two outputs: "output" and "loss"
    x: The input data. We will find an adversarial example using this.
            - Assume that x.shape = net.blobs['data'].shape
    eps: l_{\infty} norm of the perturbation that will be generated

    Returns the adversarial example, as well as just the pertubation
        (adversarial example - original input)
    """

    shape_label = net.blobs['label'].data.shape
    dummy_label = np.zeros(shape_label)

    net.blobs['data'].data[0,:,:,:] = np.squeeze(x)
    net.blobs['label'].data[...] = dummy_label

    net.forward()
    net_prediction = net.blobs['output'].data[0].argmax(axis=0).astype(np.uint32)
    net.blobs['label'].data[...] = net_prediction

    data_diff = net.backward(diffs=['data'])
    grad_data = data_diff['data']
    signed_grad = np.sign(grad_data) * eps

    adversarial_x = x + signed_grad
    return adversarial_x, signed_grad


def IterativeFGSM(net, x, eps, num_iters=-1, alpha=1, do_stop_max_pert=False):
    r"""Iterative FGSM.
       net: The caffe net. See the docstring for "fgsm" for the assumptions
       x: The input image
       eps: l_{\infty} norm of the perturbation
       num_iters: The number of iterations to run for. If it is negative, the formula
         used from Kurakin et al. Adversarial Machine Learning at Scale ICLR 2016 is used
       do_stop_max_pert: If this is true, the optimisation runs until either the max-norm 
         constraint is reached, or num_iters is reached.
    """

    clip_min = x - eps
    clip_max = x + eps

    if num_iters <= 0:
        num_iters = np.min([eps + 4, 1.25*eps]) # Used in Kurakin et al. ICLR 2016
        num_iters = int(np.max([np.ceil(num_iters), 1]))

    adversarial_x = x
    shape_label = net.blobs['label'].data.shape
    dummy_label = np.zeros(shape_label)
    net.blobs['label'].data[...] = dummy_label

    for i in range(num_iters):
        net.blobs['data'].data[0,:,:,:] = np.squeeze(adversarial_x)
        net.forward()

        net_prediction = net.blobs['output'].data[0].argmax(axis=0).astype(np.uint32)
        if i == 0:
            net.blobs['label'].data[...] = net_prediction

        data_diff = net.backward(diffs=['data'])
        grad_data = data_diff['data']

        signed_grad = np.sign(grad_data) * alpha
        adversarial_x = np.clip(adversarial_x + signed_grad, clip_min, clip_max)
        adv_perturbation = adversarial_x - x

        if do_stop_max_pert:
            max_pert = np.max(np.abs(adv_perturbation))
            if max_pert >= eps: # Due to floating point inaccuracies, need >= instead of just ==
                print "Stopping after {} iterations: Max norm reached".format(i+1)
                break

    return adversarial_x, adv_perturbation


def IterativeFGSMLeastLikely(net, x, eps, num_iters=-1, alpha=1, do_stop_max_pert=False):
    r"""Iterative FGSM Least Likely.
       This attack was proposed in Kurakin et al. Adversarial Machine Learning at Scale. ICLR 2016.
       net: The caffe net. See the docstring for "fgsm" for the assumptions
       x: The input image
       eps: l_{\infty} norm of the perturbation
       num_iters: The number of iterations to run for. If it is negative, the formula
         used from Kurakin et al. is used.
       do_stop_max_pert: If this is true, the optimisation runs until either the max-norm 
         constraint is reached, or num_iters is reached.
    """

    clip_min = x - eps
    clip_max = x + eps

    if num_iters <= 0:
        num_iters = np.min([eps + 4, 1.25*eps]) # Used in Kurakin et al. ICLR 2016
        num_iters = int(np.max([np.ceil(num_iters), 1]))

    adversarial_x = x
    shape_label = net.blobs['label'].data.shape
    dummy_label = np.zeros(shape_label)

    for i in range(num_iters):
        net.blobs['data'].data[0,:,:,:] = np.squeeze(adversarial_x)
        net.blobs['label'].data[...] = dummy_label
        net.forward()

        net_predictions = np.argsort(-net.blobs['output'].data[0], axis=0)
        target_idx = net_predictions.shape[0] - 1
        target = net_predictions[target_idx]
        target = np.squeeze(target)

        net.blobs['label'].data[...] = target

        grads = net.backward(diffs=['data'])
        grad_data = grads['data']

        signed_grad = np.sign(grad_data) * alpha
        adversarial_x = np.clip(adversarial_x - signed_grad, clip_min, clip_max)
        adv_perturbation = adversarial_x - x

        if do_stop_max_pert:
            max_pert = np.max(np.abs(adv_perturbation))
            if max_pert >= eps: # Due to floating point inaccuracies, need >= instead of just ==
                print "Stopping after {} iterations: Max norm reached".format(i+1)
                break

    return adversarial_x, adv_perturbation


def fgsm_targetted(net, x, eps, target_idx):
    r"""Targetted FGSM attack.
       net: The caffe net. See the docstring for "fgsm" for the assumptions
       x: The input image
       eps: l_{\infty} norm of the perturbation
       target_idx: The class that the adversarial attack is targetted for,
                   Note, that this is not the class id, but rather the relative ranking (0 indexed.
                   In other words, target_idx=1 means that the target will be the class
                   that was predicted with the second highest confidence.
    """

    shape_label = net.blobs['label'].data.shape
    dummy_label = np.zeros(shape_label)

    net.blobs['data'].data[0,:,:,:] = np.squeeze(x)
    net.blobs['label'].data[...] = dummy_label

    net.forward()

    net_predictions = np.argsort(-net.blobs['output'].data[0], axis=0)

    if (target_idx < 0 or target_idx > net_predictions.shape[0]):
        raise ValueError("Target idx should be an integer in the range [0,num_classes-1]")

    target = net_predictions[target_idx]
    target = np.squeeze(target)

    net.blobs['label'].data[...] = target
    grads = net.backward(diffs=['data'])
    grad_data = grads['data']

    signed_grad = np.sign(grad_data) * eps
    adversarial_x = x - signed_grad

    return adversarial_x, -signed_grad
