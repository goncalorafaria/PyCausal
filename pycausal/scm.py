from .core import *
from .stats import independence
import networkx as nx
import matplotlib.pyplot as plt
import queue
from copy import copy
import numpy as np
import pickle
import sys
from scipy.stats._distn_infrastructure  import rv_frozen

def Variable(name, dist, shape=[1]):
    arv = SourceRandomVariable(name, dist, shape)
    SCM.model.addVariable(arv)
    return arv

def HiddenVariable(name,dist, shape=[1]):
    arv = SourceRandomVariable(name, dist, shape, observed=False)
    SCM.model.addVariable(arv)
    return arv

class SCM(Named):

    def __init__(self,
                 name):

        super(SCM, self).__init__(name="SCM/"+name)
        self.nodes = {}
        self.ancestors = set([])

        self.fix()## makes this the current model.

    def fix(scm):
        SCM.model = scm

    def fix(self):
        SCM.model = self

    def addVariable(self, v):

        self.nodes[v.name] = v
        self.ancestors.add(v.name)

    def addAuxVariable(self, v):
        self.nodes[v.name] = v

    def __str__(self):
        ls = [ n.name + ":" + str(n.observed) +
              '\n::|' + str([a.name for a in n.outbound]) + '\n'
              for n in self.nodes.values() ]

        return self.name + ":: "+ str(ls)

    def mark(self, name, rv):
        # gives a special status to a variable which is the result of a computation.
        assert isinstance(rv,AuxiliaryVariable), " Can't mark an Ancestor Random Variable"

        rv._mark()
        del self.nodes[rv.name]
        #rv.name = "M/" + name
        rv.name = name
        self.addAuxVariable(rv)

        return rv

    def draw_complete(self):
        plt.figure(figsize=(12,8))
        plt.title(self.name)
        G = nx.DiGraph()

        for n in self.nodes.values():
            G.add_node(n.name)

        for n in self.nodes.values():
            for to in n.outbound :
                G.add_edge(n.name,to.name)

        nx.draw(G,with_labels=True, arrows=True, node_size=1200)

        plt.show()

    def reach(rv):
        l= []

        for n in rv.outbound :
            if n.observed :
                l.append(n)
            else :
                l = l + SCM.reach(n)

        return l

    def sample_cached(self, cache ,size):

        for n in self.nodes.values():
            if not n in cache :
                cache = n.sampling_cached(cache,size)

        return cache

    def _sample(self,size):
        results = { n[0].name : n[1]
                   for n in filter(lambda rv: rv[0].observed , self.sample(size).items()) }
        return results

    def sample(self,size):
        return self.sample_cached({}, size)

    def __invert__(self):
        return self.sample

    def __and__(self, given):
        return InterventionalConstruct(name="i:"+self.name, func=self.intervention, given=given)

    def save(self,path):
        with open(path,"wb") as fp:
            pickle.dump(self,fp)

    def load(path):
        with open(path,"rb") as fp:
            w = pickle.load(fp)

        return w

    def intervention(self, rvs, size=1):

        vs = {}
        for k,v in rvs.items():

            if isinstance(v, (int, float, complex)):
                vs[k] = np.ones(k.shape)*v

            elif isinstance(v, rv_frozen):
                onlines = v.rvs(size).reshape([size,1])
                vs[k] = (onlines,)
            else:
                vs[k] = v

        rvs = vs
        cache = {}

        for k, v in rvs.items():

            if type(v) is tuple:
                cache[k] = v[0]
            else:
                cache[k] = np.tile(v.reshape([1]+list(v.shape)),[size]+len(v.shape)*[1])

        #results = { n[0] : n[1] for n in
        #          filter(lambda rv: rv[0].observed , self.sample_cached(cache,size).items())  }
        #results = { n[0] : n[1] for n in self.sample_cached(cache,size).items() }
        return self.sample_cached(cache,size)

    def _intervention(self, rvs, size=1):

        results = { n[0].name : n[1] for n in
            filter(lambda rv: rv[0].observed , self.intervention(rvs,size).items())  }

        return results

    def draw(self, figsize=(12,8), node_size=1200 ):
        plt.figure(figsize=figsize)
        plt.title(self.uname())
        G = self.build_causal_graph()
        nx.draw(G,with_labels=True, arrows=True, node_size=node_size)
        plt.show()


    def build_causal_graph(self):
        G = nx.DiGraph()

        q = queue.Queue()

        for n in self.ancestors:
            node = self.nodes[n]
            if node.observed :
                q.put(node)

        for n in self.nodes.values():
            if n.observed :
                G.add_node(n.name)

        tested = set([])

        while not q.empty() :
            rv = q.get()
            tested.add(rv)

            lrv = rv.reach()

            for n in lrv:
                G.add_edge(rv.name, n.name)

                if n not in tested :
                    q.put(n)

        return G

    def getMarked(self):
        return list(
            map( lambda x : x.name,
                filter( lambda x : x.observed ,self.nodes.values())
            )
        )

class InterventionalConstruct(Named):
    def __init__(self,
                 name,
                 func,
                 given={}):
        super(InterventionalConstruct, self).__init__(name)
        self.given = given
        self.func = func
        self.sh = False

    def __call__(self, size):
        if self.sh :
            return { k.name: v for k,v in self.func(self.given, size).items() }
        else:
            return self.func(self.given, size)

    def __invert__(self):
        return self.__call__

    def preety(self):
        self.sh = True
        return self

    def __and__(self, ngiven):
        d = copy(self.given)
        for k,v in ngiven.items():
            d[k] = v

        return InterventionalConstruct(name="i:"+self.name, func=self.func, given=d)

class RandomVariable(Named):
    def __init__(self,
                 name,
                 observed):
        super(RandomVariable, self).__init__(name)
        self.outbound = set([])
        self.observed = observed
        self.color = "black"

    def setColor(self, color):
        self.color = color

    def getColor(self):
        return self.color

    def addChildren(self, chlds):
        self.outbound = self.outbound.union(set(chlds))

    def _mark(self):
        self.observed = True

    def __add__(self,rv):#+
        return add(self,rv)

    def __radd__(self,rv):#+
        return add(self,rv)

    def __sub__(self,rv):#-
        return subtract(self,rv)

    def __rsub__(self,rv):#-
        return subtract(rv, self)

    def __neg__(self):
        return negative(self)

    def __invert__(self):
        return self.sample

    def __lshift__(self, name):#<<
        return self.mark(name)

    def __mul__(self,rv):# *
        if isinstance(rv,RandomVariable) :
            return multiply(self,rv)
        else:
            return scale(rv,self)

    def __rmul__(self,rv):# *
        if isinstance(rv, RandomVariable) :
            return multiply(self,rv)
        else:
            return scale(rv,self)

    def __matmul__(self, rv):
        return matmul(self,rv)

    def __rmatmul__(self, rv):
        return matmul(rv,self)

    def __pow__(self,rv):# **
        return power(self,rv)

    def __truediv__(self,rv):#/
        if isinstance(rv,RandomVariable) :
            return divide(self,rv)
        else:
            return scale(1/rv, self)

    def mark(self):
        self._mark()
        return self

    def reach(self):
        l= []

        for n in self.outbound :
            if n.observed :
                l.append(n)
            else :
                l = l + n.reach()
        return l

    def getSources(self):
        l= []

        for n in self.inbound :
            if isinstance(n,SourceRandomVariable) :
                l.append(n)
            else :
                l = l + n.getSources()
        return l


    def conditional_independent_of(self, rv, given, size=500, significance=0.05):
        cache = self.sampling_cached( given, size)
        cache = rv.sampling_cached(cache, size)
        me = cache[self]
        other = cache[rv]

        return independence(me, other, significance)

    def independent_of(self, rv, size=500, significance=0.05):
        return self.conditional_independent_of(rv,{},size,significance)

    def __or__(self, value):
        return self.independent_of(value)

class AuxiliaryVariable(RandomVariable):
    def __init__(self,
                 name,
                 op,
                 inbound,
                 shape):
        super(AuxiliaryVariable, self).__init__(name, False)

        self.op = op
        self.inbound = inbound
        self.shape = shape

    def mark(self, name):
        return SCM.model.mark(name,self)

    def sample(self, size=1):
        return self.sampling_cached({} ,size)[self]

    def conditional_sampling(self, given, size):
        cache = {}

        for k, v in given.items():
            cache[k] = np.tile(v.reshape([1]+list(v.shape) ),[size]+len(v.shape)*[1])

        return self.sampling_cached(cache, size)[self]

    def sampling_cached( self, rvs, size ):
        if self in rvs.keys():
            return rvs
        z = {}
        z = {**z, **rvs}

        l=[]

        for n in self.inbound:
            d = n.sampling_cached(z, size)
            z = {**z, **d}
            l.append(z[n])

        r = self.op._apply(l)

        z[self]=r

        return z

class SourceRandomVariable(RandomVariable):##SourceRandomVariable
    def __init__(self,
                 name,
                 sampler,
                 shape,
                 observed=True):
        super(SourceRandomVariable,self).__init__(name,
                                            observed)
        self.sampler = sampler
        self.shape = list(shape)

    def sample(self, size=1):
        shp = [size]+self.shape
        return np.array(self.sampler.rvs(shp).reshape(shp))

    def conditional_sampling(self, given, size):
        if self in given.keys():
            return np.tile(given[self].reshape([1]+self.shape), [size]+len(self.shape)*[1])
        else:
            return self.sample(size)

    def sampling_cached(self, rvs, size):
        if self in rvs.keys():
            return rvs
        else:
            r = self.sample(size)
            rvs[self]=r
            return rvs

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

    def __call__(self, rvar):
        scm = SCM.model
        nrvar = AuxiliaryVariable(
                "transform/"+ self.name +
                "w/" + rvar.name+"//id:" + str(self),
                self,
                [rvar],
                shape=rvar.shape)

        rvar.addChildren([nrvar])

        scm.addAuxVariable(nrvar)

        return nrvar

    def _apply(self, tensors):
        try:
            return self.function(tensors[0])
        except FloatingPointError:
            print("Numeric error sampling. - " + str(self.function), file=sys.stderr)
            raise

class BinaryOperation(Operation):
    def __init__(self, name, function):

        super(BinaryOperation, self).__init__(name)
        self.function = function

    def __call__(self, rvar1, rvar2):

        scm = SCM.model
        nrvar = AuxiliaryVariable("combine/"+
                              self.name + "w/" + rvar1.name + "&&" + rvar2.name +
                              "//id:" + str(self),
                              self,
                              [rvar1,rvar2],
                              shape=rvar1.shape)

        rvar1.addChildren([nrvar])
        rvar2.addChildren([nrvar])

        scm.addAuxVariable(nrvar)
        return nrvar

    def _apply(self, tensors):
        try:
            return self.function(tensors[0], tensors[1])
        except FloatingPointError:
            print("Numeric error sampling. -"+ str(self.function), file=sys.stderr)
            raise

## Function definitions.

def exp(nrv):
    op = UnitaryOperation("exp",np.exp)
    return op.__call__(nrv)

def log(nrv):
    op = UnitaryOperation("log",np.log)
    return op.__call__(nrv)

def reduce_sum(nrv, axis=None, keepdims=True):
    f = lambda rv: np.sum(rv, axis=axis, keepdims=keepdims)
    op = UnitaryOperation("sum",f)
    return op.__call__(nrv)

def reduce_mean(nrv, axis=None, keepdims=True):
    f = lambda rv: np.mean(rv, axis=axis, keepdims=keepdims)
    op = UnitaryOperation("mean",f)
    return op.__call__(nrv)

def reduce_prod(nrv, axis=None, keepdims=True):
    f = lambda rv: np.prod(rv, axis=axis, keepdims=keepdims)
    op = UnitaryOperation("prod",f)
    return op.__call__(nrv)

def negative(nrv):
    op = UnitaryOperation("neg",np.negative)
    return op.__call__( nrv)

def sqrt(nrv):
    op = UnitaryOperation("sqrt",np.sqrt)
    return op.__call__( nrv)

def square(nrv):
    op = UnitaryOperation("square",np.square)
    return op.__call__( nrv)

def tanh(nrv):
    op = UnitaryOperation("tanh",np.tanh)
    return op.__call__(nrv)

def relu(nrv):
    return func(
        lambda x : np.maximum(0,x)
        )(nrv)

def leakyrelu(nrv):
    return func(
        lambda x : np.where(x > 0, x, x*(-0.25))  
        )(nrv)


def func(f,name="custom"):
    op = UnitaryOperation(name,f)
    return lambda rv: op.__call__(rv)

def power(nrv, n):
    if isinstance(n, RandomVariable) :
        op = BinaryOperation("pow", np.power)
        return op.__call__(nrv, n)
    else:
        f = lambda rv: np.power(rv,n)
        op = UnitaryOperation("power", f)
        return op.__call__(nrv)

def sin(nrv):
    op = UnitaryOperation("sin",np.sin)
    return op.__call__( nrv)

def sigmoid(nrv):
    op = UnitaryOperation("sigmoid",lambda x: 1 / (1 + np.exp(-x)))
    return op.__call__(nrv)

def cos(nrv):
    op = UnitaryOperation("cos",np.cos)
    return op.__call__( nrv)

def tan(nrv):
    op = UnitaryOperation("tan",np.tan)
    return op.__call__( nrv)

def scale(a,b):
    if isinstance(a, RandomVariable):
        rv = a
        s = b
    else:
        rv = b
        s = a

    f = lambda rv: np.multiply(s, rv)
    op = UnitaryOperation("scale",f)
    return op.__call__(rv)

def add(nrv, nrv2):
    if isinstance(nrv, RandomVariable) and isinstance(nrv2,RandomVariable):
        op = BinaryOperation("add", np.add)
        return op.__call__( nrv, nrv2)
    elif isinstance(nrv, RandomVariable):
        f = lambda rv: np.add(rv, nrv2)
        rv = nrv
    else:
        f = lambda rv: np.add(nrv, rv)
        rv = nrv2

    op = UnitaryOperation("cadd",f)
    return op.__call__(rv)

def subtract(nrv, nrv2):
    return add(nrv, negative(nrv2))

def multiply(nrv, nrv2):
    op = BinaryOperation("mul", np.multiply)
    return op.__call__( nrv, nrv2)

def inverse(nrv):
    f = lambda rv: np.divide(1.0, rv)
    op = UnitaryOperation("inverse", f)
    return op.__call__(nrv)

def matmul(nrv, nrv2):
    if isinstance(nrv, RandomVariable) and isinstance(nrv2,RandomVariable):
        op = BinaryOperation("matmul", np.matmul)
        return op.__call__( nrv, nrv2)
    elif isinstance(nrv, RandomVariable):
        f = lambda rv: np.matmul(rv, nrv2)
        rv = nrv
    else:
        f = lambda rv: np.matmul(nrv, rv)
        rv = nrv2

    op = UnitaryOperation("cmatmul",f)
    return op.__call__(rv)

def divide(nrv, nrv2):
    if isinstance(nrv, RandomVariable) and isinstance(nrv2,RandomVariable):
        op = BinaryOperation("div", np.divide)
        return op.__call__( nrv, nrv2)
    elif isinstance(nrv, RandomVariable):
        f = lambda rv: np.divide(rv, nrv2)
        rv = nrv
    else:
        f = lambda rv: np.divide(nrv, rv)
        rv = nrv2

    op = UnitaryOperation("cdiv",f)
    return op.__call__(rv)
