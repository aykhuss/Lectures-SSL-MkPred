#!/usr/bin/env python
import lhapdf
import math
import cmath
import numpy as np
import scipy
class Parameters(object):
    """very simple class to manage Standard Model Parameters"""

    #> conversion factor from GeV^{-2} into picobarns [pb]
    GeVpb = 0.3893793656e9

    def __init__(self, **kwargs):
        #> these are the independent variables we chose:
        #>  *  sw2 = sin^2(theta_w) with the weak mixing angle theta_w
        #>  *  (MZ, GZ) = mass & width of Z-boson
        self.sw2  = kwargs.pop("sw2", 0.22289722252391824808)
        self.MZ   = kwargs.pop("MZ", 91.1876)
        self.GZ   = kwargs.pop("GZ", 2.495)
        self.sPDF = kwargs.pop("sPDF", "NNPDF31_nnlo_as_0118_luxqed")
        self.iPDF = kwargs.pop("iPDF", 0)
        if len(kwargs) > 0:
            raise RuntimeError("passed unknown parameters: {}".format(kwargs))
        #> we'll cache the PDF set for performance
        lhapdf.setVerbosity(0)
        self.pdf = lhapdf.mkPDF(self.sPDF, self.iPDF)
        #> let's store some more constants (l, u, d = lepton, up-quark, down-quark)
        self.Ql = -1.;    self.I3l = -1./2.;  # charge & weak isospin
        self.Qu = +2./3.; self.I3u = +1./2.;
        self.Qd = -1./3.; self.I3d = -1./2.;
        self.alpha = 1./132.2332297912836907
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
    def vq(self, qid: int) -> float:
        if qid == 1:  # down-type
            return (self.I3d-2*self.Qd*self.sw2)/(2.*self.sw*self.cw)
        if qid == 2:  # up-type
            return (self.I3u-2*self.Qu*self.sw2)/(2.*self.sw*self.cw)
        raise RuntimeError("vq called with invalid qid: {}".format(qid))
    def aq(self, qid: int) -> float:
        if qid == 1:  # down-type
            return self.I3d/(2.*self.sw*self.cw)
        if qid == 2:  # up-type
            return self.I3u/(2.*self.sw*self.cw)
        raise RuntimeError("aq called with invalid qid: {}".format(qid))
    def Qq(self, qid: int) -> float:
        if qid == 1:  # down-type
            return self.Qd
        if qid == 2:  # up-type
            return self.Qu
        raise RuntimeError("Qq called with invalid qid: {}".format(qid))
    #> the Z-boson propagator
    def propZ(self, s: float) -> complex:
        return s/(s-complex(self.MZ**2,self.GZ*self.MZ))
#> we immediately instantiate an object (default values) in global scope
PARAM = Parameters()

def L_yy(shat: float, par=PARAM) -> float:
    return (2./3) * (par.alpha/shat) * par.Ql**2
def L_ZZ(shat: float, par=PARAM) -> float:
    return (2./3.) * (par.alpha/shat) * (par.vl**2+par.al**2) * abs(par.propZ(shat))**2
def L_Zy(shat: float, par=PARAM) -> float:
    return (2./3.) * (par.alpha/shat) * par.vl*par.Ql * par.propZ(shat).real
def H0_yy(shat: float, qid: int, par=PARAM) -> float:
    return 16.*math.pi * 3. * par.alpha*shat * par.Qq(qid)**2
def H0_ZZ(shat: float, qid: int, par=PARAM) -> float:
    return 16.*math.pi * 3. * par.alpha*shat * (par.vq(qid)**2+par.aq(qid)**2)
def H0_Zy(shat: float, qid: int, par=PARAM) -> float:
    return 16.*math.pi * 3. * par.alpha*shat * par.vq(qid)*par.Qq(qid)
def cross_partonic(shat: float, qid: int, par=PARAM) -> float:
    return (1./2./shat) * (1./36.) * (
            L_yy(shat, par) * H0_yy(shat, qid, par)
        +   L_ZZ(shat, par) * H0_ZZ(shat, qid, par)
        +2.*L_Zy(shat, par) * H0_Zy(shat, qid, par)
    )
def diff_cross(Ecm: float, Mll: float, Yll: float, par=PARAM) -> float:
    xa = (Mll/Ecm) * math.exp(+Yll)
    xb = (Mll/Ecm) * math.exp(-Yll)
    s = Ecm**2
    shat = xa*xb*s
    lum_dn = (
          par.pdf.xfxQ(+1, xa, Mll) * par.pdf.xfxQ(-1, xb, Mll)  # (d,dbar)
        + par.pdf.xfxQ(+3, xa, Mll) * par.pdf.xfxQ(-3, xb, Mll)  # (s,sbar)
        + par.pdf.xfxQ(+5, xa, Mll) * par.pdf.xfxQ(-5, xb, Mll)  # (b,bbar)
        + par.pdf.xfxQ(-1, xa, Mll) * par.pdf.xfxQ(+1, xb, Mll)  # (dbar,d)
        + par.pdf.xfxQ(-3, xa, Mll) * par.pdf.xfxQ(+3, xb, Mll)  # (sbar,s)
        + par.pdf.xfxQ(-5, xa, Mll) * par.pdf.xfxQ(+5, xb, Mll)  # (bbar,b)
        ) / (xa*xb)
    lum_up = (
          par.pdf.xfxQ(+2, xa, Mll) * par.pdf.xfxQ(-2, xb, Mll)  # (u,ubar)
        + par.pdf.xfxQ(+4, xa, Mll) * par.pdf.xfxQ(-4, xb, Mll)  # (c,cbar)
        + par.pdf.xfxQ(-2, xa, Mll) * par.pdf.xfxQ(+2, xb, Mll)  # (ubar,u)
        + par.pdf.xfxQ(-4, xa, Mll) * par.pdf.xfxQ(+4, xb, Mll)  # (cbar,c)
        ) / (xa*xb)
    return par.GeVpb * (2.*Mll/Ecm**2) * (
         lum_dn * cross_partonic(shat, 1, par)
        +lum_up * cross_partonic(shat, 2, par)
        )
if __name__ == "__main__":
    Ecm = 8e3
    for Yll in np.linspace(-3.6, 3.6, 100):
        dsig = scipy.integrate.quad(lambda M: diff_cross(Ecm,M,Yll), 80., 100., epsrel=1e-3)
        print("#Yll {:e} {:e} {:e}".format(Yll,dsig[0],dsig[1]))
    for Mll in np.linspace(10, 200, 200):
        dsig = scipy.integrate.quad(lambda Y: diff_cross(Ecm,Mll,Y), -3.6, +3.6, epsrel=1e-3)
        print("#Mll {:e} {:e} {:e}".format(Mll,dsig[0],dsig[1]))
    tot_cross = scipy.integrate.nquad(lambda M,Y: diff_cross(Ecm,M,Y), [[80.,100.],[-3.6,+3.6]], opts={'epsrel':1e-3})
    print("#total {} pb".format(tot_cross[0]))
