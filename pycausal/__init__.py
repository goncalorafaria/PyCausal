from .scm import SCM, Variable, HiddenVariable, placeholder
from .math import *
import pycausal.distributions as stats
from .inference import binary_causal_discovery 

from numpy import seterr
seterr(all='raise')
