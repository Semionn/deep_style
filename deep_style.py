#!/usr/bin/env python

import os
import argparse
import numpy as np
import scipy.misc
import sys
script_path = os.path.realpath(__file__)
sys.path.insert(1, script_path + '/caffe')
sys.path.insert(1, script_path + '/deeppy')
sys.path.insert(1, script_path + '/cudarray')
import deeppy as dp
import caffe


from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from IPython.display import clear_output, Image, display

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


def run():
    parser = argparse.ArgumentParser(
        description='Neural artistic style. Generates an image by combining '
                    'the subject from one image and the style from another.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--subject', required=True, type=str,
                        help='Subject image.')
    parser.add_argument('--style', required=True, type=str,
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
    parser.add_argument('--iterations', default=500, type=int,
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
    parser.add_argument('--vgg19', default='imagenet-vgg-verydeep-19.mat',
                        type=str, help='VGG-19 .mat file.')
    args = parser.parse_args()

    main_run(args)
    
def main_run(args):
    
    if args is None:
        temp = {
                "subject"          :  "images/margrethe2.jpg",
                "style"            :  "images/groening2.jpg",
                "output"           :  "out.png",
                "init"             :  None,
                "init_noise"       :  0.0,
                "random_seed"      :  None,
                "animation"        :  "animation",
                "iterations"       :  500,
                "learn_rate"       :  2.0,
                "smoothness"       :  5e-8,
                "subject_weights"  :  [(9, 1)],
                "style_weights"    :  [(0, 1), (2, 1), (4, 1), (8, 1), (12, 1)],
                "subject_ratio"    :  2e-2,
                "pool_method"      :  'avg',
                "vgg19"            :  'imagenet-vgg-verydeep-19.mat',
                }
        args = argparse.Namespace()
        for key in temp:
            setattr(args, key, temp[key])
    
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
    
    prototxt = "VGG_ILSVRC_19_layers_deploy.prototxt"
    params_file = "VGG_ILSVRC_19_layers.caffemodel"
    #model = caffe.io.caffe_pb2.NetParameter()
    #text_format.Merge(open(net_fn).read(), model)
    #model.force_backward = True
    #open(prototxt, 'w').write(str(model))
    
    net_caffe = caffe.Classifier(prototxt, params_file,
                           mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                           channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB
    #print [method for method in dir(net_caffe)] # if callable(getattr(net_caffe, method))
    #print net_caffe.layers
    
    def showarray(a, fmt='jpeg'):
        a = np.uint8(np.clip(a, 0, 255))
        f = StringIO()
        PIL.Image.fromarray(a).save(f, fmt)
        display(Image(data=f.getvalue()))

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
        # apply normalized ascent step to the input image
        src.data[:] += step_size/np.abs(g).mean() * g
    
        src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
                
        if clip:
            bias = net.transformer.mean['data']
            src.data[:] = np.clip(src.data, -bias, 255-bias)    
        
    def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, 
                  end='conv3_3', clip=True, **step_params):
        # prepare base images for all octaves
        octaves = [preprocess(net, base_img)]
        for i in xrange(octave_n-1):
            octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
        
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
                showarray(vis)
                print octave, i, end, vis.shape
                clear_output(wait=True)
                
            # extract details produced on the current octave
            detail = src.data[0]-octave_base
        # returning the resulting image
        return deprocess(net, src.data[0])
    
    #def learn_net_style(net, img, img_style):  
    #    return StyleNetwork(net, to_bc01(init_img), to_bc01(subject_img),
    #                       to_bc01(style_img), subject_weights, style_weights,
    #                       args.smoothness)
    
    img = np.float32(PIL.Image.open('images/margrethe.jpg'))
    #showarray(img)
    
    img_style = np.float32(PIL.Image.open(args.style))
    
    layers, img_mean = vgg19_net(args.vgg19, pool_method=args.pool_method)

    _=deepdream(net_caffe, img)
    #learn_net_style(net_caffe, img, img_style)
    #net_caffe.crop_dims
    
    return locals()
    #============================================================
    #============================================================
    #============================================================    
    layers, img_mean = vgg19_net(args.vgg19, pool_method=args.pool_method)

    # Inputs
    pixel_mean = np.mean(img_mean, axis=(0, 1))
    style_img = imread(args.style) - pixel_mean
    subject_img = imread(args.subject) - pixel_mean
    if args.init is None:
        init_img = subject_img
    else:
        init_img = imread(args.init) - pixel_mean
    noise = np.random.normal(size=init_img.shape, scale=np.std(init_img)*1e-1)
    init_img = init_img * (1 - args.init_noise) + noise * args.init_noise

    # Setup network
    subject_weights = weight_array(args.subject_weights) * args.subject_ratio
    style_weights = weight_array(args.style_weights)
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

dict_var = main_run(None)
#if __name__ == "__main__":
#    run()
