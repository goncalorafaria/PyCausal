from __future__ import division
#from .math import UnitaryOperation
from .scm import *
from .stats import independence

from copy import deepcopy
from .models import MDN, GMM
from sklearn.neural_network import MLPRegressor
import torch

import matplotlib.pyplot as plt

def fit_conditional_and_test(X_train, Y_train):

    Y_train = Y_train.ravel()
    regressor = MLPRegressor(
	                hidden_layer_sizes=(100,100, 100),
	                activation="relu",
	                max_iter=1000)
    regressor.fit(X_train,Y_train)

    residuals = (regressor.predict(X_train) - Y_train).reshape(-1,1)

    return (independence(X_train,residuals)), regressor

def binary_causal_discovery(X,Y,nameX,nameY,graphname="Sample Graph"):
    bX , mX = fit_conditional_and_test(X,Y)
    bY , mY = fit_conditional_and_test(Y,X)

    if bX and not bY :
        origin=nameX
        dest=nameY
        f=mX
    elif bY and not bX :
        origin=nameY
        dest=nameX
        f=mY
    else :
        return None

    model = SCM(graphname)
    Xvar = placeholder(origin)

    op = UnitaryOperation("universal approximator",f.predict)
    Yvar = op(Xvar).mark(dest)

    return model

class ProposalSCM():
    def __init__(self, modelX, model, lr, finetune=10, method="EM"):
        self.omodel = model
        self.model = deepcopy(model)
        self.omodelX = modelX
        self.modelX = deepcopy(modelX)
        self.optim = torch.optim.Adam(
            list(self.model.parameters()) + list(self.modelX.parameters()), lr=lr)
        self.optimX = torch.optim.Adam(
            self.modelX.parameters(),
            lr=lr)
        self.accumulation = torch.zeros((1,1))
        self.lr = lr
        self.method=method
        self.iit = finetune

    def copy(self):
        self.model = deepcopy(self.omodel)
        self.modelX = deepcopy(self.omodelX)
        self.accumulation = torch.zeros((1,1))
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr)
        self.optimX = torch.optim.Adam(
            self.modelX.parameters(),
            lr=self.lr)
    def fit(self,X_train,Y_train):

        for i in range(self.iit):
            x_h = self.modelX.forward(X_train)
            energyx = GMM.loss(x_h, X_train,
                entropy_reg = True,
                loss_type=self.method)

            self.optimX.zero_grad()
            energyx.backward()
            self.optimX.step()


        for i in range(self.iit):

            y_h = self.model.forward(X_train)
            energyyx = MDN.loss(y_h, Y_train,
                    entropy_reg = True,
                    loss_type=self.method)

            self.optim.zero_grad()
            energyyx.backward()
            self.optim.step()

        r = -energyyx.detach() - energyx.detach()
        self.accumulation += r

        return r

    def evaluate(self,X_train,Y_train):
        yt_h = self.model.forward(X_train)
        energy = MDN.loss(yt_h, Y_train,entropy_reg = False, loss_type=self.method)

        xt_h = self.modelX.forward(X_train)
        energyx = GMM.loss(xt_h, X_train, entropy_reg = False, loss_type=self.method)

        energy = energyyx + energyx

        r = -energy.detach()
        self.accumulation += r

        return r

    def online_likelihood(self):
        #print("acc!")
        #print(self.accumulation)
        return self.accumulation

def meta_objective(transfer,
         features,
         labels,
         modelxx,
         modelxy,
         modelyy,
         modelyx,
         steps=15,
         episodes=300,
         lr = 1e-3,
         metalr = 1e-2,
         finetune=10):

    tpaths = {}

    scmxy = ProposalSCM(modelxx, modelxy, lr, finetune)
    scmyx = ProposalSCM(modelyy, modelyx, lr, finetune)

    ## setup meta model.
    gamma = torch.nn.parameter.Parameter(torch.zeros((1,1)))

    optimizer = torch.optim.Adam(
              [ gamma ], lr=metalr)

    gpath = []

    for e in range(episodes):
        ## setup new model
        scmxy.copy()
        scmyx.copy()

        tpaths[e] = []

        ## prepare dataset
        batch = steps
        dt = transfer._sample(batch)

        #Xt_train = torch.tensor(dt[features][steps*batch:], dtype=torch.float32)
        #Yt_train = torch.tensor(dt[labels][steps*batch:], dtype=torch.float32)

        X_train = torch.tensor(dt[features].reshape(-1,1), dtype=torch.float32)
        Y_train = torch.tensor(dt[labels].reshape(-1,1), dtype=torch.float32)

        energyxy = scmxy.fit(X_train, Y_train)
        energyyx = scmyx.fit(Y_train, X_train)

        energy = energyxy - energyyx
        tpaths[e].append(energy.numpy())

        pb = gamma.sigmoid()

        regret = - torch.log( 1e-20 + pb * scmxy.online_likelihood().exp() + (1 - pb) * scmyx.online_likelihood().exp() )
        #print("regret")
        #print(regret)

        optimizer.zero_grad()
        regret.backward()
        optimizer.step()

        tpaths[e] = np.stack(tpaths[e])

        gpath.append(pb.detach().numpy())

    return tpaths, np.array(gpath).ravel()

def binary_causal_inference_with_interventions(
            base,
            transfer,
            A,
            B,
            epochs=1000,
            steps=15,
            episodes=100,
            lr=1e-3,
            metalr=1e-2,
            finetune=30):

    modelxx = GMM(10)
    lossxx = modelxx.fit(base, A, loss_type="EM", entropy_reg=True, epochs=epochs)
    modelxy = MDN([1,36],10)
    lossxy = modelxy.fit(base, A, B, loss_type="EM", reg=True, epoch=epochs)

    modelyy = GMM(10)
    lossyy = modelyy.fit(base, B, loss_type="EM", entropy_reg=True, epochs=epochs)
    modelyx = MDN([1,36],10)
    lossyx = modelyx.fit(base, B, A, loss_type="EM", reg=True, epoch=epochs)

    print( ( np.array(lossxy) + np.array(lossxx) )[-1])
    print( ( np.array(lossyx) + np.array(lossyy) )[-1])

    _, g = meta_objective(transfer, A, B, modelxx, modelxy, modelyy, modelyx,
                lr=lr, metalr=metalr, episodes=episodes,
                steps=steps, finetune=finetune)

    plt.plot(g, linewidth=2,c="black")
    plt.plot(np.ones(episodes),label="A->B",linestyle='dashed',c="grey")
    plt.plot(np.zeros(episodes),label="B->A",linestyle='dashed',c="black")
    plt.legend()
    plt.show()

    return g[-1]
