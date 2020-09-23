from pycausal import *

model = SCM("Simple Causal Graph")
X = Variable("X", stats.norm(loc=0,scale=1))
Z = Variable("Z", stats.beta(0.5,0.5))

Ny = HiddenVariable("Ny", stats.norm(loc=0,scale=1))

Y = Ny * Z + exp( X**2 ) << "Y"

model.draw()

print([ v.shape for v in model.sample(42).values()])
print(model.conditional_sampling({X: np.array([0])},2))
print(Y.conditional_sampling({X: np.array([0])},2))
print(model.sample(10)[NyZ])

s = model.sample(200)
print(s[Y].shape)
print("are X and Z independent? ")
#print(independence(s[X], s[Z]))
print(X.independent_of(Z))


print("are X and Y independent? ")
#print(independence(s[X], s[Y]))
print(X.independent_of(Y))

### intervene 
model = SCM("Raw I Causal Graph")
iX = Variable("iX", stats.norm(loc=0,scale=1))
P = placeholder("P")

X = iX + P << "X"

Z = Variable("Z", stats.beta(0.5,0.5))

Ny = HiddenVariable("Ny", stats.norm(loc=0,scale=1))

Y = Ny * Z +  exp(X**2) << "Y"

mi = model.intervene({P : HiddenVariable("Perturbation", stats.norm(loc=2,scale=1))})
print( mi.sample(10)["Y"].shape )
#nimodel.draw()

