from __future__ import division
import numpy as np
from scipy.stats import gamma
from sklearn.neural_network import MLPRegressor


def hsic_gam(X, Y, alph = 0.5):
    """
    X, Y are numpy vectors with row - sample, col - dim
    alph is the significance level
    auto choose median to be the kernel width
    """
    def rbf_dot(pattern1, pattern2, deg):
        size1 = pattern1.shape
        size2 = pattern2.shape

        G = np.sum(pattern1*pattern1, 1).reshape(size1[0],1)
        H = np.sum(pattern2*pattern2, 1).reshape(size2[0],1)

        Q = np.tile(G, (1, size2[0]))
        R = np.tile(H.T, (size1[0], 1))

        H = Q + R - 2* np.dot(pattern1, pattern2.T)

        H = np.exp(-H/2/(deg**2))

        return H
    n = X.shape[0]

    # ----- width of X -----
    Xmed = X

    G = np.sum(Xmed*Xmed, 1).reshape(n,1)
    Q = np.tile(G, (1, n) )
    R = np.tile(G.T, (n, 1) )

    dists = Q + R - 2* np.dot(Xmed, Xmed.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n**2, 1)

    width_x = np.sqrt( 0.5 * np.median(dists[dists>0]) )
    # ----- -----

    # ----- width of X -----
    Ymed = Y

    G = np.sum(Ymed*Ymed, 1).reshape(n,1)
    Q = np.tile(G, (1, n) )
    R = np.tile(G.T, (n, 1) )

    dists = Q + R - 2* np.dot(Ymed, Ymed.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n**2, 1)

    width_y = np.sqrt( 0.5 * np.median(dists[dists>0]) )
    # ----- -----

    bone = np.ones((n, 1), dtype = float)
    H = np.identity(n) - np.ones((n,n), dtype = float) / n

    K = rbf_dot(X, X, width_x)
    L = rbf_dot(Y, Y, width_y)

    Kc = np.dot(np.dot(H, K), H)
    Lc = np.dot(np.dot(H, L), H)

    testStat = np.sum(Kc.T * Lc) / n

    varHSIC = (Kc * Lc / 6)**2

    varHSIC = ( np.sum(varHSIC) - np.trace(varHSIC) ) / n / (n-1)

    varHSIC = varHSIC * 72 * (n-4) * (n-5) / n / (n-1) / (n-2) / (n-3)

    K = K - np.diag(np.diag(K))
    L = L - np.diag(np.diag(L))

    muX = np.dot(np.dot(bone.T, K), bone) / n / (n-1)
    muY = np.dot(np.dot(bone.T, L), bone) / n / (n-1)

    mHSIC = (1 + muX * muY - muX - muY) / n

    al = mHSIC**2 / varHSIC
    bet = varHSIC*n / mHSIC

    thresh = gamma.ppf(1-alph, al, scale=bet)[0][0]

    return (testStat, thresh)

def independence(x, y, alpha=0.05):
    testStat, thresh = hsic_gam(x,y,alpha)
    return testStat < thresh

def fit_conditional_and_test(X_train, Y_train):

Y_train = Y_train.ravel()                                                                                                                  regressor=MLPRegressor(hidden_layer_sizes=(100,100, 100),                                                                            activation="relu",                                                                                            max_iter=1000                                                                                              )
     regressor.fit(X_train,Y_train)

          residuals = (regressor.predict(X_train) - Y_train).reshape(-1,1)

               return (independence(X_train,residuals)), regressor

