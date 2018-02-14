# -*- coding: utf-8 -*-

# automatically compile Cython files
import numpy as np

import pyximport
pyximport.install(setup_args={"include_dirs" : np.get_include()})

from .channel import channel, modulate
from .decoder import *
from .encoder import encoder, get_generating_matrix
from .primitives import *
from .structures import SPMatrix
