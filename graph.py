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


N = nn.MLP(3, [4, 4, 1])
inp = Tensor(numpy.random.randint(-10, 10, (3,)))
oup = [1.0, 0.5, -2.4, 1]
yp = [N(x) for x in inp]

loss = sum((yout - ygt)**2 for ygt, yout in zip(oup, yp))
loss.backward()
res = draw_dot(loss)
res.render('gout')
