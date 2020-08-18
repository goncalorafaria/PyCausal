from pycausal import *
from scipy import stats

model = SCM("Test Graph")
X = Variable("X", stats.norm(loc=10,scale=1))

print("|testing unitary operators.")
for op in [exp, log, negative, sqrt, square, sin, cos, tan]:
    op(X).mark(str(op))

power(X,3).mark(str(power)+"3")

scale(2,X)

print("|testing binary operators.")
for op in [power, add, subtract, multiply, divide]:
    op(X,X).mark(str(op))

print("|testing ops during sampling.")
for test in model.sample(2).keys():
    if not "RV" in test.name :
        print(test.name)
        print("--\\"+test.name.split("at 0x")[0].split(" ")[1])

print("|terminated with sucess.")
