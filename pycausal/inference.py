from __future__ import division
#from .math import UnitaryOperation
from .scm import *
from .stats import independence

from copy import deepcopy
from .models import MDN
from sklearn.neural_network import MLPRegressor
import torch

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
    def __init__(self, model, lr):
        self.omodel = model
        self.model = deepcopy(model)
        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=lr)
        self.accumulation = torch.zeros((1,1)) 
        self.lr = lr
    
    def copy(self):
        self.model = deepcopy(self.omodel)
        self.accumulation = torch.zeros((1,1))
        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=self.lr)
        
    def step(self,X_train,Y_train):
        y_h = self.model.forward(X_train)
        energy = MDN.loss(y_h, Y_train,entropy_reg=False,loss_type="EM")
        self.optim.zero_grad()
        energy.backward()
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optim.step()
        
        r = -energy.detach()
        self.accumulation += r
        return r
        
    def evaluate(self,X_train,Y_train):
        yt_h = self.model.forward(X_train)
        energy = (-1) * MDN.loss(yt_h, Y_train,entropy_reg=False,loss_type="MAP")
        
        r = energy.detach()
        self.accumulation += r
        
        return r
    
    def online_likelihood(self):
        return self.accumulation

def meta_objective(transfer, 
         features, 
         labels, 
         modelxy, 
         modelyx, 
         steps=15,
         episodes=300,
         lr = 1e-3,
         metalr = 1e-2,
         val=300):
    
    tpaths = {}
    
    scmxy = ProposalSCM(modelxy,lr)
    scmyx = ProposalSCM(modelyx,lr)
    
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
        batch = 4 
        dt = transfer._sample(steps*batch+val)
        
        Xt_train = torch.tensor(dt[features][steps*batch:], dtype=torch.float32)
        Yt_train = torch.tensor(dt[labels][steps*batch:], dtype=torch.float32)
    
        for i in range(steps):

            X_train = torch.tensor(dt[features][i*batch:(i+1)*batch].reshape(-1,1), dtype=torch.float32)
            Y_train = torch.tensor(dt[labels][i*batch:(i+1)*batch].reshape(-1,1), dtype=torch.float32)
            
            energyxy = scmxy.step(X_train, Y_train)
            energyyx = scmyx.step(Y_train, X_train)

           # eval
            
           # energyxy = scmxy.evaluate(Xt_train, Yt_train)
           # energyyx = scmyx.evaluate(Yt_train, Xt_train)
            
            energy = energyxy - energyyx
            tpaths[e].append(energy.numpy())
         
        
        
        pb = gamma.sigmoid()
        #pb1, pb2 = F.logsigmoid(gamma), F.logsigmoid(-gamma)

        #logsumexp( pb1 +scmxy.online_likelihood(),  pb2 + scmyx.online_likelihood() )
       # print("ll")        
       # print(scmxy.online_likelihood())
       # print(scmxy.online_likelihood())
                
        regret = - torch.log( pb * scmxy.online_likelihood().exp() + (1 - pb) * scmyx.online_likelihood().exp() )
       # print("regret")
       # print(regret)
        
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
            epochs=2000,
            steps=15, 
            episodes=100, 
            lr=1e-3, 
            metalr=1e-2,
            val=300):
    
    modelxy = MDN([1,36],10)
    lossxy = modelxy.fit(base, A, B, epoch=epochs)

    modelyx = MDN([1,36],10)
    lossyx = modelyx.fit(base, B, A, epoch=epochs)

    _, g = meta_objective(transfer,A,B, modelxy, modelyx, 
                   lr=lr, metalr=metalr, episodes=episodes,
                   steps=steps, val=val)

    return g[-1]

 
