from .scm import *
from .distributions import norm

def linear_norm(name):

    Ax = HiddenVariable("A"+name, norm(loc=2,scale=1))
    Bx = HiddenVariable("B"+name, norm(loc=3,scale=1))
    x = HiddenVariable(name, norm(loc=0,scale=1))

    X = Ax*x + Bx

    return X

def pack(model, opaque, X, Y, Z, adj_matrix):
    pars = [ (k,v) for k,v in model.nodes.items() if "A" == k[0] or "B" == k[0] ] 

    given = { v: float((~v)(1)) for k,v in pars }

    if opaque:
        return (model&given).preety()
    else:
        return model&given, [X,Y,Z], adj_matrix

def NormalCollider(opaque=True):
    model = SCM("Simple Normal Collider")

    Axz = HiddenVariable("Axz", norm(loc=1,scale=4))
    Ayz = HiddenVariable("Ayz", norm(loc=3,scale=4))

    X = linear_norm("x") << "X"
    
    Y = linear_norm("y") << "Y"

    Z = Axz*X + Ayz*Y + linear_norm("z") << "Z"

    adj_matrix = np.array([[0,0,1], [0,0,1], [0,0,0]])

    return pack(model, opaque, X, Y, Z, adj_matrix)

def NormalChain(opaque=True):
    model = SCM("Simple Normal Chain")

    Ayz = HiddenVariable("Ayz", norm(loc=2,scale=4))
    Axy = HiddenVariable("Axy", norm(loc=5,scale=4))

    X = linear_norm("x") << "X"
    
    Y = Axy*X + linear_norm("y") << "Y"

    Z = Ayz*Y + linear_norm("z") << "Z"

    adj_matrix = np.array([[0,1,0], [0,0,1], [0,0,0]])

    return pack(model, opaque, X, Y, Z, adj_matrix)

def NormalFork(opaque=True):
    model = SCM("Simple Normal Fork")

    Axz = HiddenVariable("Axz", norm(loc=2,scale=4))
    Axy = HiddenVariable("Axy", norm(loc=0,scale=4))

    X = linear_norm("x") << "X"
    
    Y = Axy*X + linear_norm("y") << "Y"

    Z = Axz*X + linear_norm("z") << "Z"

    adj_matrix = np.array([[0,1,1], [0,0,0], [0,0,0]])

    return pack(model, opaque, X, Y, Z, adj_matrix)

def NormalMediator(opaque=True):
    model = SCM("Simple Normal Fork")

    Axz = HiddenVariable("Axz", norm(loc=1,scale=4))
    Axy = HiddenVariable("Axy", norm(loc=3,scale=4))
    Ayz = HiddenVariable("Ayz", norm(loc=0,scale=4))

    X = linear_norm("x") << "X"
    
    Y = Axy*(X) + linear_norm("y") << "Y"

    Z = Ayz*(Y) + Axz*(X) + linear_norm("z") << "Z"

    adj_matrix = np.array([[0,1,1], [0,0,1], [0,0,0]])

    return pack(model, opaque, X, Y, Z, adj_matrix)


    
