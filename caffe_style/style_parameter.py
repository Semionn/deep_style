import numpy as np


class StyleParameter:
    def __init__(self, img):
        self.image = img
        self._array = None
        self._grad_array = None

    @property
    def array(self):
        return self._array

    def _setup(self, shape):
        if self._array is None:
            self._array = np.array(self.image)
        else:
            if isinstance(shape, int):
                shape = (shape,)
            if self._array.shape != shape:
                raise ValueError('Shape %s does not match existing shape %s' %
                                 (shape, self._array.shape))

    @property
    def grad_array(self):
        if self._grad_array is None:
            self._grad_array = np.zeros_like(self.array)
        return self._grad_array

    def step(self, step):
        self._array += step
