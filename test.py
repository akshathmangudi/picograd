import picograd
from picograd.tensor import Tensor
from picograd.nn import Functions as F

a = Tensor(2.0)
b = Tensor(3.0)
a + b
a - b
a * b
b**2

# Performing operations
z = 2 * a + 3 * b
q = F.relu(z) + z * a
h = (q + 2) * z + F.relu(z)
y = h * q + b * z
y.backward()
