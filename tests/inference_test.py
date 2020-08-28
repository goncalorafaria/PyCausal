from pycausal import *
#from pycausal.inference import binary_causal_discovery, independence, fit_conditional_and_test
#from pycausal.math import UnitaryOperation

## Dificult model.

modelhard = SCM(" X <- Z -> Y Graph ")
Z = Variable("Confounder", stats.beta(0.5,0.5))
Nx = HiddenVariable("Nx", stats.norm(loc=0,scale=1))
Ny = HiddenVariable("Ny", stats.norm(loc=0,scale=1))

X = multiply(Z, Nx).mark("X")

Y = add( square(X), multiply(Ny,Z) ).mark("Y")

#modelhard.draw()

## Easy model.

modeleasy = SCM(" X -> Y Graph ")

Ny = HiddenVariable("Ny", stats.norm(loc=0,scale=1))
X = Variable("X", stats.norm(loc=0,scale=1))

Y = add( square(X), Ny).mark("Y")

#modeleasy.draw()

#model.draw_complete()

#plt.hist(Z.sample(200),bins=50)
#plt.show()

models = [modeleasy,modelhard]

for i in range(6):
    trial = 40 * 2**(5-i)
    print("test dificulty level:" + str(i+1) + " - examples " + str(trial) )
    for j in [0,1]:
        data = models[j]._sample(trial)
        scm = binary_causal_discovery(data["Y"],data["X"],"Y","X")
        
        #print(scm)
       
        if j == 0:
            if scm is None:
                print("easy model failed.")
            else:
                print("easy model passed.")
        else:
            if scm is None:
                print("hard model passed.")
            else:
                print("hard model failed.")
