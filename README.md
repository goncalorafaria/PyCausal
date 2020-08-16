
### Example

```python
from scm import *
import tensorflow_probability as tfp
import ops as math

model = SCM("Simple Causal Graph")

X = Variable("X", tfp.distributions.Normal(loc=0,scale=1))
Y = math.exp( math.square(X) ).mark("Y")
Z = math.add(X, Y).mark("Z")

model.draw()
```
#### Output
![alt text](https://github.com/goncalorafaria/PyCausal/blob/master/cimg.png)
