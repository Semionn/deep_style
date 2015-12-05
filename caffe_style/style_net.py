import numpy as np
import caffe
import cudarray as ca
from style_parameter import StyleParameter
from math import ceil
from debug_logger import Logger

def gram_matrix(img_bc01):
    n_channels = img_bc01.shape[1]
    # change reshape to numpy
    feats = np.reshape(img_bc01, (n_channels, -1))
    # feats = ca.reshape(img_bc01, (n_channels, -1))
    featsT = feats.T
    gram = np.dot(feats, featsT)
    return gram


def weight_array(weights):
    array = np.zeros(19)
    for idx, weight in weights:
        array[idx] = weight
    norm = np.sum(array)
    if norm > 0:
        array /= norm
    return array


class StyleNet(caffe.Net):
    def __init__(self, prototxt, params_file, subject_img, style_img, subject_weights, style_weights, subject_ratio,
                 layers=None, init_img=None, mean=None, channel_swap=None, smoothness=0.0, init_noise=0.0):

        caffe.Net.__init__(self, prototxt, params_file, caffe.TEST)
        self.logger = Logger("caffe_style", True)

        self.input_name = self._blob_names[0]

        # configure pre-processing
        in_ = self.inputs[0]
        self.transformer = caffe.io.Transformer(
            {in_: (1,) + tuple(np.roll(subject_img.shape, 1))})
        self.transformer.set_transpose(in_, (2, 0, 1))
        if mean is not None:
            self.transformer.set_mean(in_, mean)
        if channel_swap is not None:
            self.transformer.set_channel_swap(in_, channel_swap)

        # the reference model operates on images in [0,255] range instead of
        # [0,1]
        self.transformer.set_raw_scale(in_, 255)

        self.crop_dims = np.array(self.blobs[in_].data.shape[2:])
        self.image_dims = self.crop_dims

        if layers is None:
            layers = self.layers

        # Map weights (in convolution indices) to layer indices
        subject_weights = weight_array(subject_weights) * subject_ratio
        style_weights = weight_array(style_weights)
        self.subject_weights = np.zeros(len(layers))
        self.style_weights = np.zeros(len(layers))
        layers_len = 0
        conv_idx = 0
        for l, layer in enumerate(layers):
            if layer.type == "InnerProduct":
                break
            if layer.type == "ReLU":
                self.subject_weights[l] = subject_weights[conv_idx]
                self.style_weights[l] = style_weights[conv_idx]
                if subject_weights[conv_idx] > 0 or \
                                style_weights[conv_idx] > 0:
                    layers_len = l + 1
                conv_idx += 1

        subject_img = self.transformer.preprocess(
            self.input_name, subject_img)[np.newaxis, ...]

        style_img = self.transformer.preprocess(
            self.input_name, style_img)[np.newaxis, ...]
        init_img = subject_img
        noise = np.random.normal(
            size=init_img.shape, scale=np.std(init_img) * 1e-1)
        init_img = init_img * (1 - init_noise) + noise * init_noise

        # Discard unused layers
        layers = layers[:layers_len]
        self._layers = layers

        def output_shape(blob, x_shape, channels_n):
            b, _, img_h, img_w = x_shape
            filter_shape = (3, 3)
            padding_w = 1
            padding_h = 1
            strides_w = 1
            strides_h = 1
            out_shape = ((img_h + 2 * padding_h - filter_shape[0]) //
                         strides_h + 1,
                         (img_w + 2 * padding_w - filter_shape[1]) //
                         strides_w + 1)
            return (b, channels_n) + out_shape

        # Setup network
        x_shape = init_img.shape
        self.x = StyleParameter(init_img)
        self.x._setup(x_shape)
        self.x._array = np.array(self.x._array)

        for i, blob in enumerate(self.blobs.values()):
            blob_name = self._blob_names[i]
            if "_1" in blob_name:
                shape = blob.shape
                blob.reshape(shape[0], shape[1], x_shape[2], x_shape[3])
                x_shape = output_shape(blob, x_shape, blob.channels)
            elif "_2" in blob_name:
                shape = blob.shape
                blob.reshape(shape[0], shape[1], x_shape[2], x_shape[3])
                x_shape = (shape[0], shape[1]) + x_shape[2:]
            elif "pool" in blob_name:
                shape = blob.shape
                blob.reshape(shape[0], shape[1], x_shape[0], x_shape[1])
                x_shape = (shape[0], shape[1]) + x_shape[2:]
                x_shape = (shape[0], shape[1]) + \
                          (int(ceil(x_shape[2] / 2.0)), int(ceil(x_shape[3] / 2.0)))
            elif self.input_name in blob_name:
                shape = blob.shape
                blob.reshape(shape[0], shape[1], x_shape[2], x_shape[3])
                x_shape = output_shape(
                    blob, x_shape, self.blobs.values()[1].channels)
            print "%s : %s" % (blob_name, x_shape)

        # Precompute subject features and style Gram matrices
        self.subject_feats = [None] * len(layers)
        self.style_grams = [None] * len(layers)

        def preprocess(img):
            return np.float32(np.rollaxis(img, 2)[::-1]) - self.transformer.mean[self.input_name]

        def set_input(blob, octave):
            if len(octave[0].shape[-2:]) < 2:
                raise Exception()
            h, w = octave[0].shape[-2:]
            # old_shape = blob.shape
            blob.reshape(blob.num, blob.channels, h, w)
            blob.data[0] = octave[0]

        blobs_name_list = self.blobs.keys()

        layer_idx = 0
        curr_blob_name = blobs_name_list[layer_idx]
        next_blob_name = blobs_name_list[layer_idx + 1]

        self.blobs[curr_blob_name].data[...] = subject_img
        next_subject = self.forward(end=next_blob_name)[next_blob_name]

        _ = self.blobs[curr_blob_name].mutable_cpu_data()

        self.blobs[curr_blob_name].data[...] = style_img
        next_style = self.forward(end=next_blob_name)[next_blob_name]

        for l, layer in enumerate(layers):
            if layer.type == "InnerProduct":
                break
            if l == 0:
                continue
            if layer.type != "Convolution":
                layer_idx += 1

            if l + 1 < len(self.blobs):
                curr_blob_name = blobs_name_list[l]
                next_blob_name = blobs_name_list[l + 1]

                if "fc" not in next_blob_name and "fc" not in curr_blob_name:
                    next_subject = self.fprop(
                        curr_blob_name, next_blob_name, next_subject)
                    _ = self.blobs[curr_blob_name].mutable_cpu_data()
                    next_style = self.fprop(
                        curr_blob_name, next_blob_name, next_style)

            if self.subject_weights[l] > 0:
                curr_blob_name = blobs_name_list[layer_idx]
                result_subj = self.blobs[curr_blob_name].data
                self.subject_feats[l] = result_subj
            if self.style_weights[l] > 0:
                curr_blob_name = blobs_name_list[layer_idx]
                result_subj = self.blobs[curr_blob_name].data
                print l, blobs_name_list[layer_idx], layer.type, list(result_subj.shape)

                gram = gram_matrix(result_subj)
                # Scale gram matrix to compensate for different image sizes
                n_pixels_subject = np.prod(result_subj.shape[2:])
                n_pixels_style = np.prod(result_subj.shape[2:])
                scale = (n_pixels_subject / float(n_pixels_style))
                self.style_grams[l] = gram * scale

        self.tv_weight = smoothness
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float_)
        kernel /= np.sum(np.abs(kernel))
        self.tv_kernel = np.array(kernel[np.newaxis, np.newaxis, ...])
        self.tv_conv = ca.nnet.ConvBC01((1, 1), (1, 1))

    def fprop(self, blob_name_start, blob_name_end, data):
        if len(data[0].shape[-2:]) < 2:
            raise Exception()
        h, w = data[0].shape[-2:]
        blob_start = self.blobs[blob_name_start]
        blob_start.reshape(blob_start.num, blob_start.channels, h, w)
        blob_start.data[...] = data[0]
        if blob_name_start == self.input_name:
            return self.forward(end=blob_name_end)[blob_name_end]
        result = self.forward(start=blob_name_start, end=blob_name_end)[blob_name_end]
        return result

    def bprop(self, blob_name_start, blob_name_end, data):
        if len(data[0].shape[-2:]) < 2:
            raise Exception()
        blob_end = self.blobs[blob_name_end]
        blob_end.diff[...] = data[0]
        if blob_name_start == self.input_name:
            result = self.backward(start=blob_name_end)
        else:
            result = self.backward(start=blob_name_end, end=blob_name_start)
        return result[blob_name_start]

    @property
    def image(self):
        return np.array(self.x.array)

    @property
    def _params(self):
        return [self.x]

    @property
    def reduced_layers(self):
        return self._layers

    def _update(self):
        blobs_name_list = self.blobs.keys()

        # Forward propagation
        next_x = self.x.array
        self.logger.trace("next_x = %s" % next_x[0][0][0][0])
        x_feats = [None] * len(self.reduced_layers)
        last_blob_name = self.blobs.keys()[0]
        blob_name = self.blobs.keys()[1]
        next_x = self.fprop(last_blob_name, blob_name, next_x)
        layer_idx = 0
        for l, layer in enumerate(self.reduced_layers):
            if l == 0:
                continue
            if layer.type == "InnerProduct":
                break
            if layer.type != "Convolution":
                layer_idx += 1

            if l + 1 < len(self.blobs):
                curr_blob_name = blobs_name_list[l]
                next_blob_name = blobs_name_list[l + 1]

                if "fc" not in next_blob_name and "fc" not in curr_blob_name and "pool5" not in next_blob_name:
                    next_x = self.fprop(curr_blob_name, next_blob_name, next_x)
                    self.logger.trace("next_x = %s" % list(next_x.shape))

            if self.subject_weights[l] > 0 or self.style_weights[l] > 0:
                curr_blob_name = blobs_name_list[layer_idx]
                result_subj = self.blobs[curr_blob_name].data
                x_feats[l] = result_subj

        # Backward propagation
        grad = np.zeros_like(next_x)
        loss = np.zeros(1)

        self.logger.trace(" ".join(map(lambda layer: layer.type, self.reduced_layers)))
        self.logger.trace("x_feats = %s" % list(map(lambda x: None if x is None else x.shape, x_feats)))
        for l, style_gram in enumerate(self.style_grams):
            if style_gram is not None:
                self.logger.trace("l = %s, style_grams = %s" % (l, style_gram.shape))

        layer_idx = 16
        for l, layer in reversed(list(enumerate(self.reduced_layers))):
            if l > 29:
                continue
            if self.subject_weights[l] > 0:
                diff = x_feats[l] - self.subject_feats[l]
                norm = np.sum(np.fabs(diff)) + 1e-8
                weight = float(self.subject_weights[l]) / norm
                grad += diff * weight
                loss += 0.5 * weight * np.sum(diff ** 2)
            if self.style_weights[l] > 0:
                diff = gram_matrix(x_feats[l]) - self.style_grams[l]
                n_channels = diff.shape[0]
                x_feat = np.reshape(x_feats[l], (n_channels, -1))
                style_grad = np.reshape(np.dot(diff, x_feat), x_feats[l].shape)
                norm = np.sum(np.fabs(style_grad))
                weight = float(self.style_weights[l]) / norm
                style_grad *= weight
                self.logger.trace("style_grad = %s" % list(style_grad.shape))
                grad += style_grad
                loss += 0.25 * weight * np.sum(diff ** 2)
            if l - 2 < len(blobs_name_list) and layer.type != "Convolution":
                layer_idx -= 1
                grad = self.bprop(blobs_name_list[layer_idx], blobs_name_list[layer_idx + 1], grad)
                self.logger.trace("blob = %s, grad = %s" % (blobs_name_list[layer_idx + 1], list(grad.shape)))

        if self.tv_weight > 0:
            x = np.reshape(self.x.array, (3, 1) + grad.shape[2:])
            tv = self.tv_conv.fprop(x, self.tv_kernel)
            tv *= self.tv_weight
            grad -= np.reshape(tv, grad.shape)

        np.copyto(self.x.grad_array, grad)
        return loss
