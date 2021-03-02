from .scm import SCM, Variable, HiddenVariable, inverse , exp, log, negative, sqrt, square, power, sin, cos, tan, scale, add, subtract, multiply, matmul, divide, reduce_sum, reduce_prod, func
#from .math import *
import pycausal.distributions as stats
import pycausal.problems as problems

from numpy import seterr

seterr(all='raise')
