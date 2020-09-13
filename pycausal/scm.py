from .core import *
from .stats import independence
import networkx as nx
import matplotlib.pyplot as plt
import queue
import numpy as np
import pickle
import sys

def Variable(name, dist, shape=[1]):
    arv = SourceRandomVariable(name, dist, shape)
    SCM.model.addVariable(arv)
    return arv

def HiddenVariable(name,dist, shape=[1]):
    arv = SourceRandomVariable(name, dist, shape, observed=False)
    SCM.model.addVariable(arv)
    return arv

def placeholder(name, shape=[1]):
    rv = Placeholder(name, shape)
    SCM.model.addVariable(rv)
    return rv

class SCM(Named):

    def __init__(self,
                 name):

        super(SCM, self).__init__(name="SCM/"+name)
        self.nodes = {}
        self.ancestors = set([])

        self.fix()

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

        assert isinstance(rv,AuxiliaryVariable), " Can't mark an Ancestor Random Variable"

        rv._mark()
        del self.nodes[rv.name]
        rv.name = "M/" + name
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
        results = { n[0].uname() : n[1]
                   for n in filter(lambda rv: rv[0].observed , self.sample(size).items()) }
        return results

    def sample(self,size):
        return self.sample_cached({}, size)
    
    def save(self,path):
        with open(path,"wb") as fp:
            pickle.dump(self,fp)
    
    def intervene(self, rvs):
        return InterventionSCM(self,rvs)

    def load(path):
        with open(path,"rb") as fp:
            w = pickle.load(fp)
        
        return w
    
    def conditional_sampling(self, rvs, size=1):

        cache = {}

        for k, v in rvs.items():
            cache[k] = np.tile(v.reshape([1]+list(v.shape)),[size]+len(v.shape)*[1])

        results = { n[0] : n[1] for n in
                   filter(lambda rv: rv[0].observed , self.sample_cached(cache,size).items())  }
        return results
    
    def _conditional_sampling(self, rvs, size=1):
        
        results = { n[0].uname() : n[1] for n in 
            filter(lambda rv: rv[0].observed , self._conditional_sampling(rvs,size).items())  }
        
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
                G.add_node(n.uname())

        tested = set([])

        while not q.empty() :
            rv = q.get()
            tested.add(rv)

            lrv = rv.reach()

            for n in lrv:
                G.add_edge(rv.uname(), n.uname())

                if n not in tested :
                    q.put(n)

        return G

    def getMarked(self):
        return list(
            map( lambda x : x.uname(),
                filter( lambda x : x.observed ,self.nodes.values())
            )
        )

class InterventionSCM(SCM):
    def __init__(self,
                 scm,
                 rvs):
    
        super(InterventionSCM, self).__init__(name="I"+scm.uname())
        self.nodes = scm.nodes
        self.ancestors = scm.ancestors
        self.rvs = rvs

    def sample(self,size=1):
        cache = {}
        for rv,val in self.rvs.items():
            if isinstance(rv,RandomVariable):
                s = val.sample(1)[0]
                cache[rv] = s
                cache[val] = s
            else:
                cache[rv] =  val
        
        return self.conditional_sampling(cache,size) 
        
  #  def _sample(self,size=1):
  #      for i in self.sample().items():
  #          print(i)

class RandomVariable(Named):
    def __init__(self,
                 name,
                 observed):
        super(RandomVariable, self).__init__("RV/"+name)
        self.outbound = set([])
        self.observed = observed

    def addChildren(self, chlds):
        self.outbound = self.outbound.union(set(chlds))

    def _mark(self):
        self.observed = True
    
    def __add__(self,rv):
        return add(self,rv)

    def __sub__(self,rv):
        return subtract(self,rv)
    def __lshift__(self, name):
        return self.mark(name)
    def __mul__(self,rv):
        if isinstance(rv,RandomVariable) :
            return multiply(self,rv)
        else:
            return scale(rv,self)

    def __pow__(self,rv):
        return power(self,rv)

    def __truediv__(self,rv):
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

    def conditional_independent_of(self, rv, given, size=500, significance=0.05):
        cache = self.sampling_cached( given, size)
        cache = rv.sampling_cached(cache, size)
        me = cache[self]
        other = cache[rv]

        return independence(me, other, significance)

    def independent_of(self, rv, size=500, significance=0.05):
        return self.conditional_independent_of(rv,{},size,significance)

class AuxiliaryVariable(RandomVariable):
    def __init__(self,
                 name,
                 op,
                 inbound):
        super(AuxiliaryVariable, self).__init__(name, False)

        self.op = op
        self.inbound = inbound

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
        self.sampler=sampler
        self.shape = shape

    def sample(self, size=1):
        return self.sampler.rvs(size).reshape([size]+self.shape)

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

class Placeholder(SourceRandomVariable):
    def __init__(self,
                 name,
                 shape):
        super(Placeholder, self).__init__(name, None,shape,True)

    
    def sample(self,size):
        msg = "Placeholders can not be directly sampled because their distribution is unknown. To sample graphical models with placeholders you have to conditionally sample and determine which values to give to the placeholders "
        print(msg, err)


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
                [rvar])

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
                              [rvar1,rvar2])

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

class UOneArgOperation(UnitaryOperation):
    def __init__(self,name,function, arg):
        super(UOneArgOperation, self).__init__(name,function)
        self.arg=arg

    def _apply(self, tensors):
        try:
            return self.function(tensors[0],self.arg)
        except FloatingPointError:
            print("Numeric error sampling. -" + str(self.function), file=sys.stderr)
            raise 


def exp(nrv):
    op = UnitaryOperation("exp",np.exp)
    return op.__call__(nrv)

def log(nrv):
    op = UnitaryOperation("log",np.log)
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

def power(nrv, n):
    if isinstance(n, RandomVariable) :
        op = BinaryOperation("pow", np.power)
        return op.__call__(nrv, n)
    else:
        op = UOneArgOperation("power", np.power, n)
        return op.__call__(nrv)

def sin(nrv):
    op = UnitaryOperation("sin",np.sin)
    return op.__call__( nrv)

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
    op =  UOneArgOperation("scale", np.multiply, s)
    return op.__call__( rv)

def add(nrv, nrv2):
    if isinstance(nrv, RandomVariable) and isinstance(nrv2,RandomVariable):
        op = BinaryOperation("add", np.add)
        return op.__call__( nrv, nrv2)
    elif isinstance(nrv, RandomVariable):
        c = nrv2
        rv = nrv
    else:
        c = nrv
        rv = nrv2
    
    op = UOneArgOperation("addconstant", np.add, c) 

    return op.__call__(rv)

def subtract(nrv, nrv2):
    return add(nrv, negative(nrv2))

def multiply(nrv, nrv2):
    op = BinaryOperation("mul", np.multiply)
    return op.__call__( nrv, nrv2)

def inverse(nrv):
    op = UOneArgOperation("inverse", np.divide, 1)
    return op.__call__(nrv)

def matmul(nrv, nrv2):
    op = BinaryOperation("matmul",np.matmul)
    return op.__call__( nrv, nrv2)

def divide(nrv, nrv2):
    op = BinaryOperation("div", np.divide)
    return op.__call__( nrv, nrv2)
