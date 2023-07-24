#!/usr/bin/env python
import math
import cmath
import numpy as np
class Parameters(object):
    """very simple class to manage Standard Model Parameters"""

    #> conversion factor from GeV^{-2} into nanobarns [nb]
    GeVnb = 0.3893793656e6

    def __init__(self, **kwargs):
        #> these are the independent variables we chose:
        #>  *  sw2 = sin^2(theta_w) with the weak mixing angle theta_w
        #>  *  (MZ, GZ) = mass & width of Z-boson
        self.sw2 = kwargs.pop("sw2", 0.223)
        self.MZ  = kwargs.pop("MZ", 91.1876)
        self.GZ  = kwargs.pop("GZ", 2.4952)
        if len(kwargs) > 0:
            raise RuntimeError("passed unknown parameters: {}".format(kwargs))
        #> let's store some more constants (l, u, d = lepton, up-quark, down-quark)
        self.Ql = -1.;
        self.I3l = -1./2.;
        self.alpha = 1./137.
        #> and some derived quantities
        self.sw = math.sqrt(self.sw2)
        self.cw2 = 1.-self.sw2  # cos^2 = 1-sin^2
        self.cw = math.sqrt(self.cw2)
    #> vector & axial-vector couplings to Z-boson
    @property
    def vl(self) -> float:
        return (self.I3l-2*self.Ql*self.sw2)/(2.*self.sw*self.cw)
    @property
    def al(self) -> float:
        return self.I3l/(2.*self.sw*self.cw)
    #> the Z-boson propagator
    def propZ(self, s: float) -> complex:
        return s/(s-complex(self.MZ**2,self.GZ*self.MZ))
#> we immediately instantiate an object (default values) in global scope
PARAM = Parameters()
def G1(s: float, par=PARAM) -> float:
  return par.Ql**2 - 2. * par.vl**2 * par.Ql * par.propZ(s).real + (par.vl**2 + par.al**2)**2 * abs(par.propZ(s))**2
def G2(s: float, par=PARAM) -> float:
  return -2. * par.al**2 * par.Ql * par.propZ(s).real + 4. * par.vl**2 * par.al**2 * abs(par.propZ(s))**2
def cross(s: float, par=PARAM) -> float:
  return par.GeVnb * par.alpha**2*math.pi/(2.*s) * (8./3.) * G1(s, par)
def AFB(s: float, par=PARAM) -> float:
  return (3./4.) * G2(s,par)/G1(s,par)
if __name__ == "__main__":
    res = []
    for Ecm in np.linspace(20, 100, 200):
        s = Ecm**2
        xs = cross(s)
        afb = AFB(s)
        print("{:e} {:e} {:e}".format(Ecm,xs,afb))
