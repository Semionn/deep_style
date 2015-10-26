import cudarray as ca
from ..wrap import nnet


try:
    from ..wrap import cudnn
    _default_impl = 'cudnn'
except:
    _default_impl = 'cudarray'


class ConvBC01(object):
    def __init__(self, padding, strides, impl=None):
        self.padding = padding
        self.strides = strides
        self.impl = _default_impl if impl is None else impl
        if self.impl == 'cudarray':
            self.conv_cudnn = None
        elif self.impl == 'cudnn':
            self.conv_cudnn = cudnn.conv_bc01_cudnn(padding, strides)
        else:
            raise ValueError('invalid implementation: %s' % self.impl)

    def fprop(self, imgs, filters, convout=None):
        b, c, img_h, img_w = imgs.shape
        f, c_filters, filter_h, filter_w = filters.shape
        if c != c_filters:
            raise ValueError('channel mismatch')
        if imgs.dtype != filters.dtype:
            raise ValueError('dtype mismatch')
        img_shape = (img_h, img_w)
        self.imgs_shape = imgs.shape
        filter_shape = (filter_h, filter_w)
        convout_shape = self.output_shape(imgs.shape, f, (filter_h, filter_w))
        if convout is None:
            convout = ca.empty(convout_shape, dtype=imgs.dtype)
        else:
            if convout.shape != convout_shape:
                raise ValueError('convout.shape does not match result')
            if convout.dtype != imgs.dtype:
                raise ValueError('dtype mismatch')
        if self.impl == 'cudarray':
            nnet._conv_bc01_matmul(
                imgs._data, filters._data, b, c, f, img_shape, filter_shape,
                self.padding, self.strides, convout._data
            )
        else:
            self.conv_cudnn.fprop(
                imgs._data, filters._data, b, c, f, img_shape, filter_shape,
                convout._data
            )

        return convout

    def bprop(self, imgs, filters, convout_d, to_filters=True, to_imgs=True,
              filters_d=None, imgs_d=None):
        if imgs is not None:
            b, c, img_h, img_w = imgs.shape
        else:
            b, c, img_h, img_w = self.imgs_shape
        if filters is not None:
            f, c, filter_h, filter_w = filters.shape
        if imgs_d is not None:
            b, c, img_h, img_w = imgs_d.shape
        b_convout, f_convout, convout_h, convout_w = convout_d.shape
        imgs_shape = self.imgs_shape
        img_shape = imgs_shape[2:]
        filter_shape = (filter_h, filter_w)

        if to_filters:
            if filters_d is None:
                filters_d = ca.empty(filters.shape, dtype=filters.dtype)
            else:
                if filters_d.shape != filters.shape:
                    raise ValueError('filters_d.shape does not match result')
                if filters_d.dtype != filters.dtype:
                    raise ValueError('dtype mismatch')
            if self.impl == 'cudarray':
                nnet._conv_bc01_matmul_bprop_filters(
                    imgs._data, convout_d._data, b, c, f, img_shape,
                    filter_shape, self.padding, self.strides, filters_d._data
                )

        if to_imgs:
            if imgs_d is None:
                imgs_d = ca.empty(imgs_shape, dtype=convout_d.dtype)
            else:
                if imgs_d.shape != imgs.shape:
                    raise ValueError('imgs_d.shape does not match result')
                if imgs_d.dtype != imgs.dtype:
                    raise ValueError('dtype mismatch')
            if self.impl == 'cudarray':
                nnet._conv_bc01_matmul_bprop_imgs(
                    filters._data, convout_d._data, b, c, f, img_shape,
                    filter_shape, self.padding, self.strides, imgs_d._data
                )

        if self.impl == 'cudnn':
            imgs_ = None if imgs is None else imgs._data
            imgs_d_ = imgs_d._data if to_imgs else None
            filters_ = filters_d._data if to_filters else None
            self.conv_cudnn.bprop(imgs_, filters._data, convout_d._data,
                                  imgs_d_, filters_)

        return filters_d, imgs_d

    def output_shape(self, imgs_shape, n_filters, filter_shape):
        b, _, img_h, img_w = imgs_shape
        out_shape = ((img_h + 2*self.padding[0] - filter_shape[0]) //
                     self.strides[0] + 1,
                     (img_w + 2*self.padding[1] - filter_shape[1]) //
                     self.strides[1] + 1)
        return (b, n_filters) + out_shape
