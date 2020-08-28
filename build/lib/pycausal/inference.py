from __future__ import division
from .math import UnitaryOperation
from .scm import *
from .stats import independence

from sklearn.neural_network import MLPRegressor


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
