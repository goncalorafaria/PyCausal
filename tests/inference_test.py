from pycausal import *
from pycausal.inference import independence
from scipy import stats

model = SCM(" X -> Y Graph ")
X = Variable("X", stats.norm(loc=0,scale=1))

Ny = HiddenVariable("Ny", stats.norm(loc=0,scale=1))

Y = add(square(X), Ny).mark("Y")

model.draw()
model.draw_complete()

data = model.sample(500)

from sklearn.neural_network import MLPRegressor

def test(X_train, Y_train):
    Y_train = Y_train.ravel()

    regressor = MLPRegressor(
        hidden_layer_sizes=(100,100, 100),
        activation="relu",
        max_iter=1000
    )
    regressor.fit(X_train,Y_train)

    residuals = (regressor.predict(X_train) - Y_train).reshape(-1,1)

    return (independence(X_train,residuals)), regressor

print(test(data[X],data[Y]))
print(test(data[Y],data[X]))
