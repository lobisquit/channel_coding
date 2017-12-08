# automatically compile Cython files
import pyximport; pyximport.install()

from .decoder import is_codeword, phi_tilde
from .encoder import encoder, get_generating_matrix
