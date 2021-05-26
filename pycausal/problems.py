from .scm import *
from .distributions import norm
import random

def linear_norm(name):

    Ax = HiddenVariable("A"+name, norm(loc=2,scale=0.1))
    Bx = HiddenVariable("B"+name, norm(loc=0,scale=8))
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
    Axy = HiddenVariable("Axy", norm(loc=3,scale=6))
    Ayz = HiddenVariable("Ayz", norm(loc=0,scale=4))

    X = linear_norm("x") << "X"

    Y = Axy*(X) + linear_norm("y") << "Y"

    Z = Ayz*(Y) + Axz*(X) + linear_norm("z") << "Z"

    adj_matrix = np.array([[0,1,1], [0,0,1], [0,0,0]])

    return pack(model, opaque, X, Y, Z, adj_matrix)


def pack_listing(model, var_list, adj_matrix):
    pars = [ (k,v) for k,v in model.nodes.items() if "A" == k[0] or "B" == k[0] ]
    given = { v: float((~v)(1)) for k,v in pars }

    return model&given, var_list, adj_matrix


def RandomLinearNormal(n=3, max_degree=3):
    model = SCM("Simple Random Network")

    adj_matrix = np.zeros((n,n),dtype=np.int)
    k = 0
    frontier=[]
    while k < n :
        Nxi = linear_norm("x"+str(k))
        input_nodes = random.sample(
            list(range(k)),
            min(
                k+1,#random.randint(0, max_degree),
                k)
            )

        nc = Nxi

        for i in input_nodes:
            adj_matrix[i,k]=1
            rv = frontier[i]
            scale = HiddenVariable("A"+Nxi.name+":"+rv.name, norm(loc=0,scale=6))
            nc += scale * rv

        nc << "X"+str(k)
        frontier.append(nc)

        k+=1

    return pack_listing(model, frontier, adj_matrix)


def RandomFourierNormal(n=3, max_degree=3):
    model = SCM("Simple Fourier Random Network")

    adj_matrix = np.zeros((n,n),dtype=np.int)
    k = 0
    frontier=[]
    while k < n :
        Nxi = linear_norm("x"+str(k))
        input_nodes = random.sample(
            list(range(k)),
            min(
                k+1,#random.randint(0, max_degree),
                k)
            )

        nc = Nxi

        for i in input_nodes:
            adj_matrix[i,k]=1
            rv = frontier[i]
            scale = HiddenVariable("A"+Nxi.name+":"+rv.name, norm(loc=0,scale=2))
            phi = HiddenVariable("B"+Nxi.name+":"+rv.name, norm(loc=0,scale=0.1))
            nc += scale * tanh( phi * rv )

        nc << "X"+str(k)
        frontier.append(nc)

        k+=1

    return pack_listing(model, frontier, adj_matrix)
