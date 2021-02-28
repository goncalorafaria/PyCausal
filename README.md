# PyCausal - Causal Inference and Reasoning in Python
Package for defining large scale Structural Causal Models(SCMs), interventions and sampling from them using JAX.

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

Y = Ny * Z + exp( X**2 ) << "Y"

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
model.intervention({ X: 0 },2)
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
equivalently, we can write
```python
( ~X )(2)
```
#### output:
```
array([ 2.64181855, 17.87651557])
```

Or do independence tests(based on samples or graphical). 

```python
Y.independent_of(Ny, significance=0.05)
```
equivalently, we can write
```python
Y |= Ny
```
#### output:
```
False
```
We can actually define custom operations using ```func```, and define matrix variables and assignments. 

```
model = SCM("Matrix assignments model")


new_op = func( jax.nn.softmax, name="softmax")

X = Variable("X", stats.uniform(-2,5), shape=[2,2])
Ny = HiddenVariable("Ny",stats.beta(0.4,0.1), shape=[1,1])

y2 = -(sin(X)*2)@np.ones(shape=[2,2])
y3 = new_op(y2)

Y = reduce_sum(y3 + Ny, axis=[-1,-2], keepdims=False) << "Y"

```

#### install :~
python3 setup.py sdist bdist_wheel

pip install .
