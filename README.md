# PyCausal - Causal Inference and Reasoning in Python
Package for defining large scale Structural Causal Models(SCMs) and sampling from them using XLA.

### Example

![alt text](https://github.com/goncalorafaria/PyCausal/blob/master/eq.PNG)

where $ N_Z, N_Y, N_X $ is the standard normal.
#### Code
```python
from pycausal import *

model = SCM("Simple Causal Graph")

X = Variable("X", stats.norm(loc=0,scale=1))
Z = Variable("Z", stats.beta(0.5,0.5))

Ny = HiddenVariable("Ny", stats.norm(loc=0,scale=1))

Y = - Ny * Z + exp( X**2 ) << "Y"

model.draw()
```
with the corresponding graphical causal model, 

![alt text](https://github.com/goncalorafaria/PyCausal/blob/master/cimg.png)

```python
model.sample(2)
```

#### output:
```
{'Z': array([0.99181398, 0.02439115]), 
 'X': array([-0.07538367,  1.69771261]), 
 'Y': array([ 2.64181855, 17.87651557]) }
```

```python
model.intervention({X: np.array([0])},2)
```

#### output:
```
{'X': array([0, 0]), 
 'Z': array([0.34692997, 0.16893219]),
 'Y': array([1.42016021, 0.86607793]) }
 ```

We can also sample specific variables instead of the full model.

```python
Y.sample(2)
```
#### output:
```
 array([ 2.64181855, 17.87651557]) 
```




#### install :~
//python3 setup.py sdist bdist_wheel

pip install .
