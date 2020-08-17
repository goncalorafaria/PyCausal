from .scm import *
import numpy as np

scm = None

class Operation(Named):

    def __init__(self,name):

        super(Operation,self).__init__(name="Operation/"+name)
        self.op = name

    def _apply(self, tensor):
        pass

    def __call(self, scm, rvars ):
        pass

    def __str__(self):
        return str(id(self))

class UnitaryOperation(Operation):
    def __init__(self,name,function):

        super(UnitaryOperation, self).__init__(name)
        self.function=function

    def __call__(self, scm, rvar):
        nrvar = AuxiliaryVariable("transform/"+ self.name + "w/" + rvar.name+"//id:" + str(self),
                                  self,
                                  [rvar])

        rvar.addChildren([nrvar])

        scm.addAuxVariable(nrvar)

        return nrvar

    def _apply(self, tensors):
        return self.function(tensors[0])

class BinaryOperation(Operation):
    def __init__(self, name, function):

        super(BinaryOperation, self).__init__(name)
        self.function = function

    def __call__(self, scm, rvar1, rvar2):

        nrvar = AuxiliaryVariable("combine/"+
                              self.name + "w/" + rvar1.name + "&&" + rvar2.name +
                              "//id:" + str(self),
                              self,
                              [rvar1,rvar2])

        rvar1.addChildren([nrvar])
        rvar2.addChildren([nrvar])

        scm.addAuxVariable(nrvar)
        return nrvar

    def _apply(self, tensors):
        return self.function(tensors[0], tensors[1])

class UOneArgOperation(Operation):
    def __init__(self,name,function, arg):
        super(UOneArgOperation, self).__init__(name,function)
        self.arg=arg

    def _apply(self, tensors):
        return self.function(tensors[0],self.arg)

### examples of operations
## unitary
def exp(nrv):
    op = UnitaryOperation("exp",np.exp)
    return op.__call__(SCM.model, nrv)

def log(nrv):
    op = UnitaryOperation("log",np.log)
    return op.__call__(SCM.model, nrv)

def negative(nrv):
    op = UnitaryOperation("neg",np.negative)
    return op.__call__(SCM.model, nrv)

def sqrt(nrv):
    op = UnitaryOperation("sqrt",np.sqrt)
    return op.__call__(SCM.model, nrv)

def square(nrv):
    op = UnitaryOperation("square",np.square)
    return op.__call__(SCM.model, nrv)

def power(nrv, n):
    if isinstance(n, RandomVariable) :
        op = BinaryOperation("pow", np.power)
        return op.__call__(SCM.model, nrv, n)
    else:
        op = UOneArgOperation("power", np.power, n)
        return op.__call__(SCM.model,nrv)

def sin(nrv):
    op = UnitaryOperation("sin",np.sin)
    return op.__call__(SCM.model, nrv)

def cos(nrv):
    op = UnitaryOperation("cos",np.cos)
    return op.__call__(SCM.model, nrv)

def tan(nrv):
    op = UnitaryOperation("tan",np.tan)
    return op.__call__(SCM.model, nrv)

def scale(a,b):
    if isinstance(a, RandomVariable):
        rv = a
        s = b
    else:
        rv = b
        s = a
    op =  UOneArgOperation("scale", np.multiply, s)
    return op.__call__(SCM.model, rv)

## binary
def add(nrv, nrv2):
    op = BinaryOperation("add", np.add)
    return op.__call__(SCM.model, nrv, nrv2)

def subtract(nrv, nrv2):
    return add(nrv, negative(nrv2))

def multiply(nrv, nrv2):
    op = BinaryOperation("mul", np.multiply)
    return op.__call__(SCM.model, nrv, nrv2)

def matmul(nrv, nrv2):
    op = BinaryOperation("matmul",np.matmul)
    return op.__call__(SCM.model, nrv, nrv2)

def divide(nrv, nrv2):
    op = BinaryOperation("div", np.divide)
    return op.__call__(SCM.model, nrv, nrv2)
