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
model.conditional_sampling({X: np.array([0])},2)
```

#### output:
```
{'X': array([0, 0]), 
 'Z': array([0.34692997, 0.16893219]),
 'Y': array([1.42016021, 0.86607793]) }
 ```

We can also sample variables instead of the full model.
#### install :~
//python3 setup.py sdist bdist_wheel

pip install .
