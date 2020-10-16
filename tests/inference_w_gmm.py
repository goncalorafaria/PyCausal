from pycausal import *
from pycausal.inference import binary_causal_inference_with_interventions
from pycausal.models import GMM

### (X^3 + X - X^2) * sin(X)
base = SCM("Observational Source Model")
X = Variable("X", stats.uniform(-2,5))
Ny = HiddenVariable("Ny",stats.norm(0,0.5))

y2 = sin(X) * (-2) * X

Y = y2 + Ny << "Y"

#base.draw()

######
interv = SCM("Interventional Source Model")
iX = HiddenVariable("iX", stats.uniform(-2,5))
P = placeholder("P")

X = P + iX << "X"

Ny = HiddenVariable("Ny",stats.norm(0,0.5))

y2 = sin(X) * (-2) * X

Y = y2 + Ny << "Y"

Perturbation = HiddenVariable("Perturbation", stats.uniform(2,1))
transfer = interv.intervene({P: Perturbation})

a = binary_causal_inference_with_interventions(
        base, transfer, "X", "Y", epochs=500, steps=15,
        episodes=100, lr=1e-2, metalr=0.5, finetune=20)

print(a)
