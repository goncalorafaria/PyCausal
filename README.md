# PyCausal - Causal Inference and Reasoning in Python
Package for defining Structural Causal Models and for Structure Identification from data.

### Example

![alt text](https://github.com/goncalorafaria/PyCausal/blob/master/eq.PNG)

where $ N_Z, N_Y, N_X $ is the standard normal.
#### Code
```python
from pycausal import *
from scipy import stats

model = SCM("Simple Causal Graph")

X = Variable("X", stats.norm(loc=0,scale=1))
Z = Variable("Z", stats.beta(0.5,0.5))

Ny = HiddenVariable("Ny", stats.norm(loc=0,scale=1))

NyZ = multiply(Ny,Z)
Y = add(NyZ, exp(square(X))).mark("Y")

model.draw()
```
with the corresponding graphical causal model, 

![alt text](https://github.com/goncalorafaria/PyCausal/blob/master/cimg.png)



#### install :~
//python3 setup.py sdist bdist_wheel

pip install .
