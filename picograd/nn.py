import numpy
from picograd.tensor import Tensor # have to figure out this path error

class Functions: 

    """
        This will contain the crux of our program. Mainly: 
            1. The activation functions (relu, tanh, sigmoid)
            2. The layers
    """

    @staticmethod
    def relu(x): 
        assert isinstance(x, Tensor), "Input should be a Tensor object" 
        out = Tensor(numpy.maximum(0, x.data), (x,), 'ReLU')

        def _backward(): 
            x.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    @staticmethod
    def tanh(x): 
        assert isinstance(x, Tensor), "Input should be a Tensor object" 
        out = Tensor(np.tanh(x.data), (x,), 'tanh')
        
        def _backward(): 
            x.grad += (1 - out.data**2) * out.grad
        out._backward = _backward
        return out 

    @staticmethod
    def sigmoid(x): 
        assert isinstance(x, Tensor), "Input shoul be a Tensor object" 
        out = Tensor(1 / (1 + np.exp(-x.data)), (x,), 'sigmoid')

        def _backward(): 
            x.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward

        return out
