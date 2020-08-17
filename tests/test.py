from pycausal import *
from scipy import stats
import numpy as np
model = SCM("Simple Causal Graph")

X = Variable("X", stats.norm(loc=0,scale=1))
Z = Variable("Z", stats.beta(0.5,0.5))
Ny = HiddenVariable("Ny", stats.norm(loc=2, scale=1))

NyZ = multiply(Ny,Z)

Y = add(square(X), NyZ).mark("Y")

model.draw()
model.draw_complete()
print(Y.sample(30))
print(Y.conditional_sampling( { X : np.array([0]) },20 ))
print(Y.sampling_cached({},1))
print("####################")
print(model.sample(2))

for k,v in model.sample(4000).items() :
    plt.figure()
    plt.hist(v , bins=80)
    plt.legend(k)
    plt.savefig("hist" + k + ".png")

print(model.getMarked())
