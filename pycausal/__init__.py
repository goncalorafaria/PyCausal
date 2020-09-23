from .scm import SCM, Variable, HiddenVariable, placeholder, inverse , exp, log, negative, sqrt, square, power, sin, cos, tan, scale, add, subtract, multiply, matmul, divide
#from .math import *
import pycausal.distributions as stats
from .inference import binary_causal_discovery 

from numpy import seterr
seterr(all='raise')
