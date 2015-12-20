#!/home/semionn/anaconda2/bin/python

# todo: replace /caffe to /distibute
# todo: script to make distibute
import numpy as np
import os
import sys

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(script_path, 'caffe'))
sys.path.insert(1, os.path.join(script_path, 'cudarray'))

import argparse
import scipy.misc
import caffe
import caffe.draw
import PIL.Image
import caffe_style.style_net as style_net
from caffe_style.style_adam_solver import StyleAdamSolver


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


def save_img(a, file_name):
    a = np.uint8(np.clip(a, 0, 255))
    PIL.Image.fromarray(a).save(file_name)


def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']


def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data']))


def run():
    parser = argparse.ArgumentParser(
        description='Neural artistic style. Generates an image by combining '
                    'the subject from one image and the style from another.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--subject', type=str, default="images/tuebingen2.jpg",
                        help='Subject image.')
    parser.add_argument('--style', type=str, default="images/starry_night2.jpg",
                        help='Style image.')
    parser.add_argument('--output', default='out.jpeg', type=str,
                        help='Output image.')
    parser.add_argument('--init', default=None, type=str,
                        help='Initial image. Subject is chosen as default.')
    parser.add_argument('--init-noise', default=0.1, type=float_range,
                        help='Weight between [0, 1] to adjust the noise level '
                             'in the initial image.')
    parser.add_argument('--random-seed', default=None, type=int,
                        help='Random state.')
    parser.add_argument('--animation', default='animation', type=str,
                        help='Output animation directory.')
    parser.add_argument('--iterations', default=150, type=int,
                        help='Number of iterations to run.')
    parser.add_argument('--learn-rate', default=3.0, type=float,
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
    parser.add_argument('--gpu', default='True', choices=['true', 'false'],
                        type=str, help='turn on/off gpu mode.')
    parser.add_argument('--vgg19', default='imagenet-vgg-verydeep-19.mat',
                        type=str, help='VGG-19 .mat file.')
    parser.add_argument('--prototxt', default='VGG_ILSVRC_19_layers_deploy.prototxt',
                        type=str, help='VGG-19 .prototxt file.')
    parser.add_argument('--caffemodel', default='VGG_ILSVRC_19_layers.caffemodel',
                        type=str, help='VGG-19 .caffemodel file.')
    parser.add_argument('--solver-params', default='solver_adam.prototxt',
                        type=str, help='Adam solver .prototxt file.')
    args = parser.parse_args()

    main_run(args)


def resize_big_image(image_path):
    maxwidth = 300
    img = PIL.Image.open(image_path)
    if img.size[0] > maxwidth:
        wpercent = (maxwidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((maxwidth,hsize), PIL.Image.ANTIALIAS)
        img.save(image_path)
    elif img.size[1] > maxwidth:
        hpercent = (maxwidth/float(img.size[1]))
        wsize = int((float(img.size[0])*float(hpercent)))
        img = img.resize((wsize, maxwidth), PIL.Image.ANTIALIAS)
        img.save(image_path)


def main_run(args):
    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    prototxt = args.prototxt
    params_file = args.caffemodel

    resize_big_image(args.subject)
    resize_big_image(args.style)

    pixel_mean = [103.939, 116.779, 123.68]
    if args.gpu == "true":
        caffe.set_mode_gpu()
        caffe.set_device(0)
    style_img = caffe.io.load_image(args.style)
    subject_img = caffe.io.load_image(args.subject)
    net_caffe = style_net.StyleNet(prototxt, params_file, subject_img, style_img,
                                   args.subject_weights, args.style_weights, args.subject_ratio,
                                   mean=np.float32(pixel_mean))
    net = net_caffe
    src = net.blobs['data']

    h, w = subject_img.shape[:2]
    src.reshape(src.num, src.channels, h, w)
    src.data[...] = net.transformer.preprocess('data', subject_img)

    params = net._params
    style_adam = StyleAdamSolver(learn_rate=args.learn_rate)
    optimization_states = [style_adam.init_state(p) for p in params]
    for i in range(args.iterations):
        cost = np.mean(net.update())
        vis = deprocess(net, src.data[0])
        save_img(vis, args.output)
        for param, state in zip(params, optimization_states):
            style_adam.step(param, state)
        print('Iteration: %i, cost: %.4f' % (i, cost))


if __name__ == "__main__":
    run()
