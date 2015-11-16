#!/usr/bin/env python

import os
import argparse
import numpy as np
import scipy.misc
import sys

#todo: replace /caffe to /distibute
#todo: script to make distibute
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(script_path,'caffe'))
sys.path.insert(1, os.path.join(script_path,'deeppy'))
sys.path.insert(1, os.path.join(script_path,'cudarray'))
import deeppy as dp
import caffe
import caffe_style.style_net as style_net

import caffe.draw
from google.protobuf import text_format
from caffe.proto import caffe_pb2


import numpy as np
import scipy.ndimage as nd
import PIL.Image
from IPython.display import clear_output

from matconvnet import vgg19_net
from style_network import StyleNetwork


def weight_tuple(s):
    try:
        conv_idx, weight = map(float, s.split(','))
        return conv_idx, weight
    except:
        raise argparse.ArgumentTypeError('weights must by "int,float"')


def float_range(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0, 1]" % x)
    return x


def weight_array(weights):
    array = np.zeros(19)
    for idx, weight in weights:
        array[idx] = weight
    norm = np.sum(array)
    if norm > 0:
        array /= norm
    return array


def imread(path):
    return scipy.misc.imread(path).astype(dp.float_)


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)


def to_bc01(img):
    return np.transpose(img, (2, 0, 1))[np.newaxis, ...]


def to_rgb(img):
    return np.transpose(img[0], (1, 2, 0))

def save_img(a, file_name='out.jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    PIL.Image.fromarray(a).save(file_name)

def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def objective_L2(dst):
    dst.diff[:] = dst.data

def make_step(net, step_size=1.5, end='inception_4c/output',
              jitter=32, clip=True, objective=objective_L2):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    print("blobs: ", net.blobs)
    dst = net.blobs[end]


    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift

    net.forward(end=end)
    objective(dst)  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    if np.abs(g).mean() == 0:
        raise Exception("fail")
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)

def deepdream(net, base_img, iter_n=40, octave_n=4, octave_scale=1.4,
              end='conv2_1', clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    #for i in xrange(octave_n-1):
    #    octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))

    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step(net, end=end, clip=clip, **step_params)

            # visualization
            vis = deprocess(net, src.data[0])
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))
            save_img(vis)
            print octave, i, end, vis.shape
            clear_output(wait=True)

        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])

def run():
    parser = argparse.ArgumentParser(
        description='Neural artistic style. Generates an image by combining '
                    'the subject from one image and the style from another.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--subject', type=str, default="images/tuebingen3.jpg",
                        help='Subject image.')
    parser.add_argument('--style', type=str, default="images/starry_night3.jpg",
                        help='Style image.')
    parser.add_argument('--output', default='out.png', type=str,
                        help='Output image.')
    parser.add_argument('--init', default=None, type=str,
                        help='Initial image. Subject is chosen as default.')
    parser.add_argument('--init-noise', default=0.0, type=float_range,
                        help='Weight between [0, 1] to adjust the noise level '
                             'in the initial image.')
    parser.add_argument('--random-seed', default=None, type=int,
                        help='Random state.')
    parser.add_argument('--animation', default='animation', type=str,
                        help='Output animation directory.')
    parser.add_argument('--iterations', default=250, type=int,
                        help='Number of iterations to run.')
    parser.add_argument('--learn-rate', default=2.0, type=float,
                        help='Learning rate.')
    parser.add_argument('--smoothness', type=float, default=5e-8,
                        help='Weight of smoothing scheme.')
    parser.add_argument('--subject-weights', nargs='*', type=weight_tuple,
                        default=[(9, 1)],
                        help='List of subject weights (conv_idx,weight).')
    parser.add_argument('--style-weights', nargs='*', type=weight_tuple,
                        default=[(0, 1), (2, 1), (4, 1), (8, 1), (12, 1)],
                        help='List of style weights (conv_idx,weight).')
    parser.add_argument('--subject-ratio', type=float, default=2e-2,
                        help='Weight of subject relative to style.')
    parser.add_argument('--pool-method', default='avg', type=str,
                        choices=['max', 'avg'], help='Subsampling scheme.')
    parser.add_argument('--gpu', default='on', choices=['on', 'off'],
                        type=str, help='turn on/off gpu mode.')
    parser.add_argument('--vgg19', default='imagenet-vgg-verydeep-19.mat',
                        type=str, help='VGG-19 .mat file.')
    parser.add_argument('--prototxt', default='VGG_ILSVRC_19_layers_deploy.prototxt',
                        type=str, help='VGG-19 .prototxt file.')
    parser.add_argument('--caffemodel', default='VGG_ILSVRC_19_layers.caffemodel',
                        type=str, help='VGG-19 .caffemodel file.')
    args = parser.parse_args()

    main_run(args)
    
def main_run(args):
    
    load_deeppy = False
    load_dream = True

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
    
    prototxt = args.prototxt
    params_file = args.caffemodel

    img_subj = np.float32(PIL.Image.open(args.subject))
    img_style = np.float32(PIL.Image.open(args.style))
    
    pixel_mean = [123.68, 103.939, 116.779]
    if load_dream:
        if args.gpu == "on":
            caffe.set_mode_gpu()
            caffe.set_device(0);
        style_img = to_bc01(imread(args.style) - pixel_mean)
        subject_img = to_bc01(imread(args.subject) - pixel_mean)
        net_caffe = style_net.StyleNet(prototxt, params_file, subject_img, style_img,
                                       args.subject_weights, args.style_weights, args.subject_ratio,
                                       mean = np.float32(pixel_mean), channel_swap = (2,1,0));

    if load_deeppy:
        style_img = imread(args.style) - pixel_mean
        subject_img = imread(args.subject) - pixel_mean
        if args.init is None:
            init_img = subject_img
        else:
            init_img = imread(args.init) - pixel_mean
        noise = np.random.normal(size=init_img.shape, scale=np.std(init_img)*1e-1)
        init_img = init_img * (1 - args.init_noise) + noise * args.init_noise
    
        subject_weights = weight_array(args.subject_weights) * args.subject_ratio
        style_weights = weight_array(args.style_weights)
        layers, img_mean = vgg19_net(args.vgg19, pool_method=args.pool_method)
        net = StyleNetwork(layers, to_bc01(init_img), to_bc01(subject_img),
                           to_bc01(style_img), subject_weights, style_weights,
                           args.smoothness)
        # Repaint image
        def net_img():
            return to_rgb(net.image) + pixel_mean

        if not os.path.exists(args.animation):
            os.mkdir(args.animation)

        params = net._params
        learn_rule = dp.Adam(learn_rate=args.learn_rate)
        learn_rule_states = [learn_rule.init_state(p) for p in params]
        for i in range(args.iterations):
            imsave(os.path.join(args.animation, '%.4d.png' % i), net_img())
            cost = np.mean(net._update())
            for param, state in zip(params, learn_rule_states):
                learn_rule.step(param, state)
            print('Iteration: %i, cost: %.4f' % (i, cost))
        imsave(args.output, net_img())

    if load_dream:
        _=deepdream(net_caffe, img_subj)
        #_ = net_caffe._update();

if __name__ == "__main__":
    run()
