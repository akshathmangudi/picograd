import numpy

class Tensor:
    """
        The "Tensor" datastructure is what will be used for all operations in this library. 

        It will contain few important operations like: 
            1. Basic operations - add, mul, pow (May 31st) -> DONE
            2. Activation functions (relu, tanh) (June 1st)
            3. Backprop implementation (June 2nd) -> DONE 

        All naming convention is with respect to the PEP documentation.
    """

    def __init__(self, data, _nodes=(), _oper=''): 
        # Initialize everything. 
        self.data = data
        self.grad = 0 
        self._backward = lambda: None
        self._prev = set(_nodes)
        self._oper = _oper # For debugging

    def __add__(self, other): 
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward(): 
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other): 
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward(): 
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out 

    def __pow__(self, other): 
        assert isinstance(other, (int, float)), "Other forms of power are not allowed"
        out = Tensor(self.data**other, (self,), f'**{other}')

        def _backward(): 
            self.grad += (other * self.data**(other-1)) * out.grad # x^n = nx^(n-1)
        out._backward = _backward

        return out

    def backward(self): 
        # This is the topological order of all the children in the graph. 
        elem = []
        visited = set()
        def build(v): 
            if v not in visited: 
                visited.add(v)
                for child in v._prev: 
                    build(child)
                elem.append(v)
        build(self)

        self.grad = 1
        for v in reversed(elem): 
            v._backward()

    # Implementing bi-directional operations (ex: a.__mul__(2) AND 2.__mul__(a))
    
    def __neg__(self): 
        return self * -1

    def __radd__(self, other): 
        return self + other

    def __sub__(self, other): 
        return self + (-other)

    def __rsub__(self, other): 
        return other + (-self)

    def __rmul__(self, other): 
        return self * other
    
    def __div__(self, other): 
        return self * other**-1

    def __rdiv(self, other): 
        return other * self**-1

    def __repr__(self): 
        return f"Tensor(data={self.data})"
    

