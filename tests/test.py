from pycausal import *
from scipy import stats

model = SCM("Simple Causal Graph")

X = Variable("X", stats.norm(loc=0,scale=1))
Z = Variable("Z", stats.beta(0.5,0.5))

Ny = HiddenVariable("Ny", stats.norm(loc=0,scale=1))

NyZ = multiply(Ny,Z)
Y = add(NyZ, exp(square(X))).mark("Y")

model.draw()

#print(model.sample(2))
print(model.conditional_sampling({X: np.array([0])},2))
print(Y.conditional_sampling({X: np.array([0])},2))
