
### Example

```python
from scm import *
import tensorflow_probability as tfp
import ops as math

model = SCM("Simple Causal Graph")

X = Variable("X", tfp.distributions.Normal(loc=0,scale=1))
Z = Variable("Z", tfp.distributions.Normal(loc=0,scale=1))

Ny = HiddenVariable("Ny", tfp.distributions.Normal(loc=0,scale=1))

NyZ = math.multiply(Ny,Z)
Y = math.add(NyZ, math.exp(math.square(X))).mark("Y")

model.draw()
```
#### Output
![alt text](https://github.com/goncalorafaria/PyCausal/blob/master/cimg.png)
