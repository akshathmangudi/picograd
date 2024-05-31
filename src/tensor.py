import numpy 

class Tensor:
    """
        The "Tensor" datastructure is what will be used for all operations in this library. 

        It will contain few important operations like: 
            1. Basic operations - add, mul, pow (May 31st)
            2. Activation functions (relu, tanh) (June 1st)
            3. Backprop implementation (June 2nd) 

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

        return out

    def __mul__(self, other): 
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        return out 

    def __pow__(self, other): 
        assert isinstance(other, (int, float)), "Other forms of power are not allowed"
        out = Tensor(self.data**other, (self,), f'**{other}')

        return out
    
    # Implementing bi-directional operations (ex: a.__mul__(2) AND 2.__mul__(a))
    
    def __neg__(self): 
        return self * -1

    def __badd__(self, other): 
        return self + other

    def __sub__(self, other): 
        return self + (-other)

    def __bisub__(self, other): 
        return other + (-self)

    def __bimul__(self, other): 
        return self * other
    
    def __div__(self, other): 
        return self * other**-1

    def __bidiv(self, other): 
        return other * self**-1

    def __repr__(self): 
        return f"Tensor(data={self.data})"
    

