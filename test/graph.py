import random
import numpy
from graphviz import Digraph
from picograd.tensor import Tensor
from picograd import nn 
from picograd.nn import Functions as F 

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        dot.node(name=str(id(n)), label = "{ data %.4f | grad %.4f }" % (n.data, n.grad), shape='record')
        if n._oper:
            dot.node(name=str(id(n)) + n._oper, label=n._oper)
            dot.edge(str(id(n)) + n._oper, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._oper)
    
    return dot

"""
n = nn._Neuron(4)
x = [Tensor(2.0), Tensor(3.0), Tensor(1.0), Tensor(2.3)]
y = n(x)
y.backward()

dot = draw_dot(y)
dot.render('neuron_visualization')

model = nn.MLP(3, [2, 1])
inp = [Tensor(2.0), Tensor(3.4)]
y = model(inp)
y.backward()

dot = draw_dot(y)
dot.render("mlp1_visualization")
"""


"""
    Now, we will visualize the MLP for the following model: 

    1. 2 neurosn in the input layer
    2. 4 neurons in the first two hidden layers 
    3. 2 neurons in the last hidden layer
    4. 1 output layer and a relu at the end
"""

model = nn.MLP(2, [4, 4, 2, 1])

x1, x2 = numpy.random.uniform(0, 5, 2)

x = [Tensor(x1), Tensor(x2)]
Y = model(x)

Y.backward()

dot = draw_dot(Y)
dot.render('binary_class')

