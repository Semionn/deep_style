import numpy as np
import itertools
from ..base import Model, ParamMixin, PhaseMixin
from ..input import Input
from ..loss import SoftmaxCrossEntropy


class NeuralNetwork(Model, PhaseMixin):
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss
        self.bprop_until = next((idx for idx, l in enumerate(self.layers)
                                 if isinstance(l, ParamMixin)), 0)
        self.layers[self.bprop_until].bprop_to_x = False
        self._initialized = False

    def _setup(self, x_shape, y_shape=None):
        # Setup layers sequentially
        if self._initialized:
            return
        for layer in self.layers:
            layer._setup(x_shape)
            x_shape = layer.y_shape(x_shape)
        self.loss._setup(x_shape, y_shape)
        self._initialized = True

    @property
    def _params(self):
        all_params = [layer._params for layer in self.layers
                      if isinstance(layer, ParamMixin)]
        # Concatenate lists in list
        return list(itertools.chain.from_iterable(all_params))

    @PhaseMixin.phase.setter
    def phase(self, phase):
        if self._phase == phase:
            return
        self._phase = phase
        for layer in self.layers:
            if isinstance(layer, PhaseMixin):
                layer.phase = phase

    def _update(self, x, y):
        self.phase = 'train'

        # Forward propagation
        y_pred = self.fprop(x)

        # Backward propagation
        grad = self.loss.grad(y_pred, y)
        for layer in reversed(self.layers[self.bprop_until:]):
            grad = layer.bprop(grad)
        return self.loss.loss(y_pred, y)

    def fprop(self, x):
        for layer in self.layers:
            x = layer.fprop(x)
        return x

    def y_shape(self, x_shape):
        for layer in self.layers:
            x_shape = layer.y_shape(x_shape)
        return x_shape

    def predict(self, input):
        """ Calculate the output for the given input x. """
        input = Input.from_any(input)
        self.phase = 'test'

        if isinstance(self.loss, SoftmaxCrossEntropy):
            # Add softmax from SoftmaxCrossEntropy
            self.layers += [self.loss]

        y = np.empty(self.y_shape(input.x.shape))
        y_offset = 0
        for batch in input.batches():
            x_batch = batch['x']
            y_batch = np.array(self.fprop(x_batch))
            batch_size = x_batch.shape[0]
            y[y_offset:y_offset+batch_size, ...] = y_batch
            y_offset += batch_size

        if isinstance(self.loss, SoftmaxCrossEntropy):
            self.layers = self.layers[:-1]
        return y
