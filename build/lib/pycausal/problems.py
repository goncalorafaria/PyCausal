from .scm import *
from .distributions import norm



def linear_norm(name):

    Ax = HiddenVariable("A"+name, norm(loc=0,scale=1))
    Bx = HiddenVariable("B"+name, norm(loc=0,scale=1))
    x = HiddenVariable(name, norm(loc=0,scale=1))

    X = Ax*x + Bx

    return X

def pack(model, opaque, X, Y, Z):
    pars = [ (k,v) for k,v in model.nodes.items() if "A" == k[0] or "B" == k[0] ] 

    given = { v: float((~v)(1)) for k,v in pars }

    if opaque:
        return (model&given).preety()
    else:
        return model&given, [X,Y,Z]

def NormalCollider(opaque=True):
    model = SCM("Simple Normal Collider")

    Axz = HiddenVariable("Axz", norm(loc=0,scale=1))
    Ayz = HiddenVariable("Ayz", norm(loc=0,scale=1))

    X = linear_norm("x") << "X"
    
    Y = linear_norm("y") << "Y"

    Z = Axz*X + Ayz*Y + linear_norm("z") << "Z"

    return pack(model, opaque, X, Y, Z)

def NormalChain(opaque=True):
    model = SCM("Simple Normal Chain")

    Ayz = HiddenVariable("Ayz", norm(loc=0,scale=1))
    Axy = HiddenVariable("Axy", norm(loc=0,scale=1))

    X = linear_norm("x") << "X"
    
    Y = Axy*X + linear_norm("y") << "Y"

    Z = Ayz*Y + linear_norm("z") << "Z"

    return pack(model, opaque, X, Y, Z)

def NormalFork(opaque=True):
    model = SCM("Simple Normal Fork")

    Axz = HiddenVariable("Axz", norm(loc=0,scale=1))
    Axy = HiddenVariable("Axy", norm(loc=0,scale=1))

    X = linear_norm("x") << "X"
    
    Y = Axy*X + linear_norm("y") << "Y"

    Z = Axz*X + linear_norm("z") << "Z"

    return pack(model, opaque, X, Y, Z)

def NormalMediator(opaque=True):
    model = SCM("Simple Normal Fork")

    Axz = HiddenVariable("Axz", norm(loc=0,scale=1))
    Axy = HiddenVariable("Axy", norm(loc=0,scale=1))
    Ayz = HiddenVariable("Ayz", norm(loc=0,scale=1))

    X = linear_norm("x") << "X"
    
    Y = Axy*X + linear_norm("y") << "Y"

    Z = Ayz*Y + Axz*X + linear_norm("z") << "Z"

    return pack(model, opaque, X, Y, Z)


    
