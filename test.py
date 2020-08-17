from scm import *
import ops as math
from scipy import stats

model = SCM("Simple Causal Graph")

X = Variable("X", stats.norm(loc=0,scale=1))
Z = Variable("Z", stats.norm(loc=0,scale=1))
Ny = HiddenVariable("Ny", stats.norm(loc=0, scale=1))

NyZ = math.multiply(Ny,Z)

Y = math.add( NyZ , math.square(X) ).mark("Y")

model.draw()
model.draw_complete()
print(Y.sample(30))
print(Y.conditional_sampling( { X : np.array([0]) },20 ))
print(Y.sampling_cached({},1))
print("####################")
print(model.sample(2))

