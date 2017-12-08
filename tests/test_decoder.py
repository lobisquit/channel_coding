from math import exp, log

import numpy as np

import LDPC


def phi_definition(x):
    # those thresholds were proposed during the lesson
    if x < 1e-5:
        return 12
    if x > 12:
        return 0
    k = exp(-x)
    return log( (1+k) / (1-k) )

def test_phi():
    x = np.logspace(-7, 15)
    def_values = [phi_definition(a) for a in x]

    # note that vector is modified in place
    LDPC.phi_tilde(x)

    for i, value in enumerate(x):
        assert def_values[i] == x[i]
