from .scm import *
from .distributions import norm, halfnorm, uniform, bernoulli
import random

def linear_norm(name):

    Ax = HiddenVariable("A"+name, uniform(1,1.4))
    Bx = HiddenVariable("B"+name, norm(loc=0,scale=0.3))
    x = HiddenVariable(name, norm(loc=0,scale=1))

    X = 0.4*x*Ax + Bx

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

"""
def RandomLinearNormal(n=3, p=0.5):
    model = SCM("Simple Random Network")

    adj_matrix = np.zeros((n,n),dtype=np.int)
    k = 1
    frontier=[linear_norm("x"+str(0))]
    active = np.zeros(n, dtype=np.bool)
    active[0]=True

    while k < n :
        include = np.logical_and(
                        p > np.random.uniform(size=(n)),
                        active
                    )

        Nxi = linear_norm("x"+str(k))
        nc = Nxi

        for i in range(include.shape[0]):
            if include[i]:
                adj_matrix[i,k]=1
                rv = frontier[i]
                scale = HiddenVariable("A"+Nxi.name+":"+rv.name, norm(loc=0,scale=6))
                nc += scale * rv

        nc << "X"+str(k)
        frontier.append(nc)
        k+=1
        active[k-1]=True

    return pack_listing(model, frontier, adj_matrix)
"""


def RandomLinearNormal(n=3, p=0.5):
    return RandomFourierNormal(n=n,p=p,transform=None)

def RandomFourierNormal(n=3, p=0.5, transform=None, dist=None, nonnormal=False):

    model = SCM("Simple Random Network")

    adj_matrix = np.zeros((n,n),dtype=np.int)
    k = 1
    frontier=[linear_norm("x"+str(0))]
    active = np.zeros(n, dtype=np.bool)
    active[0]=True
    while k < n :
        include = np.logical_and(
                p > np.random.uniform(size=(n)),
                active
            )

        if dist is None:
            Nxi = linear_norm("x"+str(k))
        else:
            Nxi = dist("x"+str(k))
        
        #Nxi *= 0.05

        inps = []
        for i in range(include.shape[0]):
            if include[i]:
                adj_matrix[i,k]=1
                rv = frontier[i]
                #scale = HiddenVariable("A"+Nxi.name+":"+rv.name, norm(loc=0,scale=2))
                inps.append(rv)
                #nc += scale * relu( rv )

        #Nxi = Nxi 
        if len(inps)>0 :
            
            if nonnormal:
                inps.append(Nxi)
                nc = mlp(inps,transform)
            else:
                print(transform)
                if transform is not None :
                    nc = mlp(inps,transform) + 0.4*Nxi
                else:
                    nc = mlp(inps,lambda x_: x_,layers=1) + 0.4*Nxi
        else :
            nc = Nxi

        nc << "X"+str(k)
        frontier.append(nc)

        k+=1
        active[k-1]=True

    return pack_listing(model, frontier, adj_matrix)

def mlp(inps,transform,layers=3,hidden=8):
    print(transform)
    if layers == 1: 

        j = 1
        vs = []
        for v in inps:
            scale = HiddenVariable("A"+str(j)+":"+"head"+v.name+":"+v.name, uniform(0.25,1))
            sign = HiddenVariable("As"+str(j)+":"+"head"+v.name+":"+v.name, bernoulli(0.5))

            vs.append((1 - 2*sign)*v*scale)

        return sum(vs)
    else:
        os = [ ]
        for j in range(hidden):
            
            vs = []
            for v in inps:
                scale = HiddenVariable("A"+str(j)+":"+str(layers)+v.name+":"+v.name, norm(0.0,1.0))
                vs.append(v*scale)
            
            scale = HiddenVariable("B"+str(j)+":"+"head"+v.name+":b"+v.name, uniform(0.05,1.5))
            sign = HiddenVariable("Bs"+str(j)+":"+"head"+v.name+":b"+v.name, bernoulli(0.5))

            vs.append((1 - 2*sign))

            os.append( transform(sum(vs)) )

        return mlp(os, transform, layers-1, hidden)

def RandomNonLinearNonNormal(n=3, p=0.5):
    transform = square

    def linear_cauchy(name):
        Ax = HiddenVariable("A"+name, norm(loc=2,scale=0.1))
        Bx = HiddenVariable("B"+name, norm(loc=0,scale=8))
        x = HiddenVariable(name,  halfnorm() )

        X = Ax*x + Bx

        return X

    return RandomFourierNormal(n=n, p=p, transform=transform, dist=linear_cauchy)


def isource(atomic):
    signal = np.random.binomial(n=1,p=0.5)*2 - 1
    value = signal * np.random.uniform(1,2)*1.5

    if atomic:
        return value 
    else:
        return norm(
            loc=value,
            scale=0.1)


def sample_perfect_intervention(
        adj_matrix,
        vars,
        n:int=1,
        atomic=False):

    indegree = adj_matrix.sum(0)
    outdegree = adj_matrix.sum(1)

    elegible_nodes = [ i for i in range(indegree.shape[0]) ]

    ints = random.sample(
            elegible_nodes,n)

    int_dist=[(i,isource(atomic)) for i in ints ]

    conditioning = { vars[i] : v for i,v in int_dist}

    zs = np.zeros(adj_matrix.shape[0],dtype=np.int)

    for i,_ in int_dist:
        zs[i]=1

    return conditioning, zs

