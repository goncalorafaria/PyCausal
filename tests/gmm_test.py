from pycausal.models.models import GMM
from pycausal import *
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

base = SCM(" Gaussian mixture")
X1 = Variable("X1",stats.norm(2,0.8))
X2 = Variable("X2",stats.norm(-1.2,1))
Z = HiddenVariable("Z",stats.bernoulli(0.6))

Y = Z * X1 + (Z*(-1) + 1) * X2 << "Y"

model = GMM(2,False)
l = model.fit(base,"Y",lr=1e-1, epochs=1000,loss_type="EM",entropy_reg=True)
plt.plot(l)

#print(model.forward(None)[1])
#model = GMM(2,False)
#l, _ = model.fit(base,"Y",lr=1e-1, epochs=3000,loss_type="EM",entropy_reg=True)
#plt.plot(l)
print(model.forward(None)[1])
plt.show()

