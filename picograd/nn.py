import numpy
import numba
import random
from .tensor import Tensor


class Functions:
    """
    This will contain the crux of our program. Mainly:
        1. The activation functions (relu, tanh, sigmoid)
        2. The layers
    """

    @staticmethod
    def relu(x):
        assert isinstance(x, Tensor), "Input should be a Tensor object"
        out = Tensor(numpy.maximum(0, x.data), (x,), "ReLU")

        def _backward():
            x.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    @staticmethod
    def tanh(x):
        assert isinstance(x, Tensor), "Input should be a Tensor object"
        out = Tensor(numpy.tanh(x.data), (x,), "tanh")

        def _backward():
            x.grad += (1 - out.data**2) * out.grad

        out._backward = _backward
        return out

    @staticmethod
    def sigmoid(x):
        assert isinstance(x, Tensor), "Input shoul be a Tensor object"
        out = Tensor(1 / (1 + numpy.exp(-x.data)), (x,), "sigmoid")

        def _backward():
            x.grad += out.data * (1 - out.data) * out.grad

        out._backward = _backward

        return out


class _Module:
    def zero_grad(self):
        for p in self.params():
            p.grad = 0

    def params(self):
        return []


class _Neuron(_Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Tensor(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Tensor(random.uniform(-1, 1))
        self.nonlin = nonlin

    def __call__(self, x):
        out = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return Functions.relu(out) if self.nonlin else out

    def params(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class _Layer(_Module):

    def __init__(self, nin, nout, **kwargs):
        self.neu = [_Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neu]
        return out[0] if len(out) == 1 else out

    def params(self):
        return [p for n in self.neu for p in n.params()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neu)}]"


class MLP(_Module):
    def __init__(self, nin, nouts):
        pair = [nin] + nouts  # We are creating a pair and we will iterate over them.
        self.layers = [
            _Layer(pair[i], pair[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def params(self):
        return [p for layer in self.layers for p in layer.params()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
