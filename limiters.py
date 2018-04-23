import numpy as np

def min3(a, b, c):
    return np.minimum(a, np.minimum(b, c))

def min2(a, b):
    return np.minimum(a, b)

def max2(a, b):
    return np.maximum(a, b)

def max3(a, b, c):
    return np.maximum(a, np.maximum(b, c))

class Limiters(object):
    def __init__(self, name):
        if name == "koren":
            self.limiter = np.vectorize(self.limiter_koren)
        elif name == "minmod":
            self.limiter = np.vectorize(self.limiter_minmod)
        elif name == "osher":
            self.limiter = np.vectorize(self.limiter_osher)
        else:
            pass
    def limiter_koren(self, r):
        return max2(0.0, min3(2.0*r, (1+2.0*r)/3.0, 2))

    def limiter_minmod(self, r):
        return max2(0, min2(1.0, r))

    def limiter_mc(self, r):
        return max2(0.0, min3(2.0*r, 0.5*(1.0 + r), 2.0))

    def limiter_osher(self, r):
        beta = 1.5
        return max2(0.0, min2(r, beta))

    def limiter_ospre(self, r):
        return 1.5*(r*r + r)/(r*r + r + 1.0)

    def limiter_superbee(self, r):
        return max3(0.0, min2(2.0*r, 1.0), min2(r, 2.0))

    def limiter_sweby(self, r):
        beta = 1.5
        return max3(0.0, min2(beta*r, 1.0), min2(r, beta))

    def limiter_vanleer(self, r):
        if r >= 0:
            return 2.0*r/(1.0 + r)
        else:
            return 0
