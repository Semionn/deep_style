import numpy as np
import cudarray as ca
from .base import PickleMixin


_FLT_MIN = np.finfo(ca.float_).tiny


class Loss(PickleMixin):
    @classmethod
    def from_any(cls, arg):
        if isinstance(arg, Loss):
            return arg
        elif isinstance(arg, str):
            if arg == 'softmaxce':
                return SoftmaxCrossEntropy()
            elif arg == 'bce':
                return BinaryCrossEntropy()
            elif arg == 'mse':
                return MeanSquaredError()
        raise ValueError('Invalid constructor arguments: %s' % arg)

    def _setup(self, pred_shape, target_shape=None):
        pass

    def loss(self, pred, target):
        """ Returns the loss calculated from the target and the input. """
        raise NotImplementedError()

    def grad(self, pred, target):
        """ Returns the input gradient. """
        raise NotImplementedError()


class SoftmaxCrossEntropy(Loss):
    """
    Softmax + cross entropy (aka. multinomial logistic loss)
    """

    def __init__(self):
        self.name = 'softmaxce'
        self._tmp_x = None
        self._tmp_y = None
        self._tmp_target = None
        self._tmp_one_hot = None
        self.n_classes = None

    def _setup(self, pred_shape, target_shape=None):
        self.n_classes = pred_shape[1]

    def _softmax(self, x):
        # caching wrapper
        if self._tmp_x is not x:
            self._tmp_y = ca.nnet.softmax(x)
            self._tmp_x = x
        return self._tmp_y

    def _one_hot(self, target):
        # caching wrapper
        if self._tmp_target is not target:
            self._tmp_one_hot = ca.nnet.one_hot_encode(target, self.n_classes)
            self._tmp_target = target
        return self._tmp_one_hot

    def loss(self, pred, target):
        pred = self._softmax(pred)
        target = self._one_hot(target)
        return ca.nnet.categorical_cross_entropy(y_pred=pred, y_true=target)

    def grad(self, pred, target):
        pred = self._softmax(pred)
        target = self._one_hot(target)
        return -(target - pred)

    def fprop(self, x):
        return ca.nnet.one_hot_decode(self._softmax(x))

    def y_shape(self, x_shape):
        return (x_shape[0],)


class BinaryCrossEntropy(Loss):
    def __init__(self):
        self.name = 'bce'

    def loss(self, pred, target):
        pred = ca.maximum(pred, _FLT_MIN)
        return -ca.mean(target*ca.log(pred) + (1 - target)*ca.log(1 - pred),
                        axis=1)

    def grad(self, pred, target):
        pred = ca.maximum(pred, _FLT_MIN)
        return -(target/pred - (1-target)/(1-pred))


class MeanSquaredError(Loss):
    def __init__(self):
        self.name = 'mse'
        self.n_feats = None

    def _setup(self, pred_shape, target_shape=None):
        self.n_feats = pred_shape[1]

    def loss(self, pred, target):
        return ca.mean((target-pred)**2, axis=1)

    def grad(self, pred, target):
        return 2.0 / self.n_feats * (pred - target)
