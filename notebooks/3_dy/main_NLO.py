#!/usr/bin/env python
import lhapdf
import math
import cmath
import numpy as np
import scipy
import vegas
from enum import IntEnum, unique

@unique
class PID(IntEnum):
  """
  define Parton Id's
  """
  GLUON = 0
  DOWN = 1
  UP = 2
  def __str__(self):
    return self.name.lower()

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
    def vq(self, qid: PID) -> float:
        if qid == PID.DOWN:
            return (self.I3d-2*self.Qd*self.sw2)/(2.*self.sw*self.cw)
        if qid == PID.UP:
            return (self.I3u-2*self.Qu*self.sw2)/(2.*self.sw*self.cw)
        raise RuntimeError("vq called with invalid qid: {}".format(qid))
    def aq(self, qid: PID) -> float:
        if qid == PID.DOWN:
            return self.I3d/(2.*self.sw*self.cw)
        if qid == PID.UP:
            return self.I3u/(2.*self.sw*self.cw)
        raise RuntimeError("aq called with invalid qid: {}".format(qid))
    def Qq(self, qid: PID) -> float:
        if qid == PID.DOWN:
            return self.Qd
        if qid == PID.UP:
            return self.Qu
        raise RuntimeError("Qq called with invalid qid: {}".format(qid))
    #> the Z-boson propagator
    def propZ(self, s: float) -> complex:
        return s/(s-complex(self.MZ**2,self.GZ*self.MZ))
#> we immediately instantiate an object (default values) in global scope
PARAM = Parameters()

def L_yy(Q2: float, par=PARAM) -> float:
    return (2./3) * (par.alpha/Q2) * par.Ql**2
def L_ZZ(Q2: float, par=PARAM) -> float:
    return (2./3.) * (par.alpha/Q2) * (par.vl**2+par.al**2) * abs(par.propZ(Q2))**2
def L_Zy(Q2: float, par=PARAM) -> float:
    return (2./3.) * (par.alpha/Q2) * par.vl*par.Ql * par.propZ(Q2).real

def H0_yy(Q2: float, qid: PID, par=PARAM) -> float:
    return 16.*math.pi * 3. * par.alpha*Q2 * par.Qq(qid)**2
def H0_ZZ(Q2: float, qid: PID, par=PARAM) -> float:
    return 16.*math.pi * 3. * par.alpha*Q2 * (par.vq(qid)**2+par.aq(qid)**2)
def H0_Zy(Q2: float, qid: PID, par=PARAM) -> float:
    return 16.*math.pi * 3. * par.alpha*Q2 * par.vq(qid)*par.Qq(qid)

def cross_partonic_LO(Q2: float, qid: PID, par=PARAM) -> float:
    return (    L_yy(Q2, par) * H0_yy(Q2, qid, par)
            +   L_ZZ(Q2, par) * H0_ZZ(Q2, qid, par)
            +2.*L_Zy(Q2, par) * H0_Zy(Q2, qid, par) )

def lumi(ida: PID, idb: PID, xa: float, xb: float, Q: float, par=PARAM) -> float:
    # return 1.
    # return 1./(xa*xb)
    if (ida,idb) == (PID.DOWN,PID.DOWN):
        return (
              par.pdf.xfxQ(+1, xa, Q) * par.pdf.xfxQ(-1, xb, Q)  # (d,dbar)
            + par.pdf.xfxQ(+3, xa, Q) * par.pdf.xfxQ(-3, xb, Q)  # (s,sbar)
            + par.pdf.xfxQ(+5, xa, Q) * par.pdf.xfxQ(-5, xb, Q)  # (b,bbar)
            + par.pdf.xfxQ(-1, xa, Q) * par.pdf.xfxQ(+1, xb, Q)  # (dbar,d)
            + par.pdf.xfxQ(-3, xa, Q) * par.pdf.xfxQ(+3, xb, Q)  # (sbar,s)
            + par.pdf.xfxQ(-5, xa, Q) * par.pdf.xfxQ(+5, xb, Q)  # (bbar,b)
            ) / (xa*xb)
    if (ida,idb) == (PID.UP,PID.UP):
        return (
              par.pdf.xfxQ(+2, xa, Q) * par.pdf.xfxQ(-2, xb, Q)  # (u,ubar)
            + par.pdf.xfxQ(+4, xa, Q) * par.pdf.xfxQ(-4, xb, Q)  # (c,cbar)
            + par.pdf.xfxQ(-2, xa, Q) * par.pdf.xfxQ(+2, xb, Q)  # (ubar,u)
            + par.pdf.xfxQ(-4, xa, Q) * par.pdf.xfxQ(+4, xb, Q)  # (cbar,c)
            ) / (xa*xb)
    if (ida,idb) == (PID.DOWN,PID.GLUON):
        return (
              par.pdf.xfxQ(+1, xa, Q) * par.pdf.xfxQ(0, xb, Q)  # (d,g)
            + par.pdf.xfxQ(+3, xa, Q) * par.pdf.xfxQ(0, xb, Q)  # (s,g)
            + par.pdf.xfxQ(+5, xa, Q) * par.pdf.xfxQ(0, xb, Q)  # (b,g)
            + par.pdf.xfxQ(+1, xa, Q) * par.pdf.xfxQ(0, xb, Q)  # (dbar,g)
            + par.pdf.xfxQ(+3, xa, Q) * par.pdf.xfxQ(0, xb, Q)  # (sbar,g)
            + par.pdf.xfxQ(+5, xa, Q) * par.pdf.xfxQ(0, xb, Q)  # (bbar,g)
            ) / (xa*xb)
    if (ida,idb) == (PID.GLUON,PID.DOWN):
        return (
              par.pdf.xfxQ(0, xa, Q) * par.pdf.xfxQ(+1, xb, Q)  # (g,d)
            + par.pdf.xfxQ(0, xa, Q) * par.pdf.xfxQ(+3, xb, Q)  # (g,s)
            + par.pdf.xfxQ(0, xa, Q) * par.pdf.xfxQ(+5, xb, Q)  # (g,b)
            + par.pdf.xfxQ(0, xa, Q) * par.pdf.xfxQ(+1, xb, Q)  # (g,dbar)
            + par.pdf.xfxQ(0, xa, Q) * par.pdf.xfxQ(+3, xb, Q)  # (g,sbar)
            + par.pdf.xfxQ(0, xa, Q) * par.pdf.xfxQ(+5, xb, Q)  # (g,bbar)
            ) / (xa*xb)
    if (ida,idb) == (PID.UP,PID.GLUON):
        return (
              par.pdf.xfxQ(+2, xa, Q) * par.pdf.xfxQ(0, xb, Q)  # (u,g)
            + par.pdf.xfxQ(+4, xa, Q) * par.pdf.xfxQ(0, xb, Q)  # (c,g)
            + par.pdf.xfxQ(-2, xa, Q) * par.pdf.xfxQ(0, xb, Q)  # (ubar,g)
            + par.pdf.xfxQ(-4, xa, Q) * par.pdf.xfxQ(0, xb, Q)  # (cbar,g)
            ) / (xa*xb)
    if (ida,idb) == (PID.GLUON,PID.UP):
        return (
              par.pdf.xfxQ(0, xa, Q) * par.pdf.xfxQ(+2, xb, Q)  # (g,u)
            + par.pdf.xfxQ(0, xa, Q) * par.pdf.xfxQ(+4, xb, Q)  # (g,c)
            + par.pdf.xfxQ(0, xa, Q) * par.pdf.xfxQ(-2, xb, Q)  # (g,ubar)
            + par.pdf.xfxQ(0, xa, Q) * par.pdf.xfxQ(-4, xb, Q)  # (g,cbar)
            ) / (xa*xb)
    raise RuntimeError("lumi called with invalid ids: ({},{})".format(ida,idb))

def diff_cross_LO(Ecm: float, Mll: float, Yll: float, par=PARAM) -> float:
    xa = (Mll/Ecm) * math.exp(+Yll)
    xb = (Mll/Ecm) * math.exp(-Yll)
    s = Ecm**2
    shat = Mll**2  # = xa*xb*s
    return par.GeVpb * (2.*Mll/Ecm**2) * (1./2./shat) * (1./36.) * (
          lumi(PID.DOWN, PID.DOWN, xa, xb, Mll, par)
          * cross_partonic_LO(shat, PID.DOWN, par)
        + lumi(PID.UP, PID.UP, xa, xb, Mll, par)
          * cross_partonic_LO(shat, PID.UP,   par)
        )

def del_qq(za : float, zb: float, facF: float = 1.) -> tuple:
    OmZa = 1-za; OpZa = 1+za; OmZa2 = OmZa*OpZa;
    OmZb = 1-zb; OpZb = 1+zb; OmZb2 = OmZb*OpZb;
    LnOmZa = math.log(OmZa); LnOpZa = math.log(OpZa); LnOmZa2 = math.log(OmZa2);
    LnOmZb = math.log(OmZb); LnOpZb = math.log(OpZb); LnOmZb2 = math.log(OmZb2);
    LnF = - 2. * math.log(facF)
    Ln2 = math.log(2.)
    RaRb = + (2*(1 + za*zb))/(OmZa*OmZb*OpZa*OpZb*za*zb) + (OmZa2*(1 + za*zb))/(OmZb*OpZb*(za**2)*((za + zb)**2)) + (OmZb2*(1 + za*zb))/(OmZa*OpZa*(zb**2)*((za + zb)**2))
    RaEb = + LnF/(2.*(za**2)) + ((1 + Ln2 + LnOmZa2 - 2*LnOpZa)*OmZa)/(2.*(za**2)) - OmZa/(2.*OmZb*(za**2)) - LnF/(2.*za) + LnF/(OmZa*za) + (Ln2 - LnOpZa)/(OmZa*za) - 1/(OmZa*OmZb*za)
    EaRb = + LnF/(2.*(zb**2)) + ((1 + Ln2 + LnOmZb2 - 2*LnOpZb)*OmZb)/(2.*(zb**2)) - OmZb/(2.*OmZa*(zb**2)) - LnF/(2.*zb) + LnF/(OmZb*zb) + (Ln2 - LnOpZb)/(OmZb*zb) - 1/(OmZa*OmZb*zb)
    EaEb = + (3*LnF)/2. - LnF/OmZa - LnF/OmZb + 1/(OmZa*OmZb) + (-8 + (math.pi**2))/2.
    return RaRb, RaEb, EaRb, EaEb

def del_qg(za : float, zb: float, facF: float = 1.) -> tuple:
    OmZa = 1-za; OpZa = 1+za; OmZa2 = OmZa*OpZa;
    OmZb = 1-zb; OpZb = 1+zb; OmZb2 = OmZb*OpZb;
    LnOmZa = math.log(OmZa); LnOpZa = math.log(OpZa); LnOmZa2 = math.log(OmZa2);
    LnOmZb = math.log(OmZb); LnOpZb = math.log(OpZb); LnOmZb2 = math.log(OmZb2);
    LnF = - 2. * math.log(facF)
    Ln2 = math.log(2.)
    qgtoq = 0.
    RaRb = + (OmZa2*(1 + za*zb))/(za*((za + zb)**3)) - (2*OmZb2*za*(1 + za*zb))/(OmZa*OpZa*zb*((za + zb)**2)) + (1 + za*zb)/(OmZa*OpZa*za*(zb*zb)*(za + zb))
    RaEb = 0.
    EaRb = + (1 + Ln2 + LnOmZb2 - 2*LnOpZb)/(2.*(zb*zb)) - 1/(2.*OmZa*(zb*zb)) - ((Ln2 + LnOmZb2 - 2*LnOpZb)*OmZb)/zb + OmZb/(OmZa*zb) - ((-LnF + qgtoq)*(1 - 2*zb + 2*(zb*zb)))/(2.*(zb*zb))
    EaEb = 0.
    return RaRb, RaEb, EaRb, EaEb

def del_db(za : float, zb: float, facF: float = 1.) -> tuple:
    #> for debugging:  multiply with cross = (xa*xb)**2 inside the integrand; lumi -> 1./(xa*xb) --> integral = 2.75
    cA   = 1.1  # 1 -->  +1/2  |  delta(1-za) delta(1-zb)
    cBa  = 1.2  # 1 -->  +1/4  |  delta(1-za)
    cBb  = 1.3  # 1 -->  +1/4  |  delta(1-zb)
    cCab = 1.4  # 1 -->  -1/2  |  delta(1-za) 1/(1-zb)_+
    cCba = 1.5  # 1 -->  -1/2  |  delta(1-zb) 1/(1-za)_+
    cD   = 1.6  # 1 -->  +1/2  |  1/(1-za)_+ 1/(1-zb)_+
    cEa  = 1.7  # 1 -->  -1/4  |  1/(1-za)_+
    cEb  = 1.8  # 1 -->  -1/4  |  1/(1-zb)_+
    cF   = 1.9  # 1 -->  +1/8  |
    RaRb = + cF/((za**2)*(zb**2)) + cEa/((1 - za)*(za**2)*(zb**2)) + cEb/((za**2)*(1 - zb)*(zb**2)) + cD/((1 - za)*(za**2)*(1 - zb)*(zb**2))
    RaEb = + cBb/(za**2) + cCba/((1 - za)*(za**2)) - cEb/((za**2)*(1 - zb)) - cD/((1 - za)*(za**2)*(1 - zb))
    EaRb = + cBa/(zb**2) - cEa/((1 - za)*(zb**2)) + cCab/((1 - zb)*(zb**2)) - cD/((1 - za)*(1 - zb)*(zb**2))
    EaEb = + cA - cCba/(1 - za) - cCab/(1 - zb) + cD/((1 - za)*(1 - zb))
    return RaRb, RaEb, EaRb, EaEb

def integrand_NLO(xa : float, xb: float, za : float, zb: float, Q: float, facR: float = 1., facF: float = 1., par=PARAM):
    if (za>=1) or (zb>=1) or (za<=0) or (zb<=0):
        return 0.
    res = 0.
    fac_NLO = (par.pdf.alphasQ(facR*Q)/math.pi) * (4./3.)
    cross_dn = cross_partonic_LO(Q**2, PID.DOWN, par)
    cross_up = cross_partonic_LO(Q**2, PID.UP, par)
    #> 0., 0., 0., 0.  #
    # RaRb_LO, RaEb_LO, EaRb_LO, EaEb_LO = del_db(za,zb,facF)  # <- debugging
    # RaRb_LO *= za*zb; RaEb_LO *= za; EaRb_LO *= zb;
    # print("RaRb_LO: {:e},  RaEb_LO: {:e},  EaRb_LO: {:e},  EaEb_LO: {:e},  ".format(RaRb_LO, RaEb_LO, EaRb_LO, EaEb_LO))
    # input()
    RaRb_LO, RaEb_LO, EaRb_LO, EaEb_LO = 0., 0., 0., 0.  # <- LO for tests
    RaRb_qq, RaEb_qq, EaRb_qq, EaEb_qq = del_qq(za,zb,facF)
    RaRb_qg, RaEb_qg, EaRb_qg, EaEb_qg = 0., 0., 0., 0.  # del_qg(za,zb,facF)
    RaRb_gq, EaRb_gq, RaEb_gq, EaEb_gq = 0., 0., 0., 0.  # del_qg(zb,za,facF)  # za <-> zb from `qg` & region swap
    if za>xa and zb>xb:
        # RaRb
        res += cross_dn * (za*zb) * (
                        RaRb_LO * lumi(PID.DOWN, PID.DOWN,  xa/za, xb/zb, Q, par) * (1./36.)
            + fac_NLO * RaRb_qq * lumi(PID.DOWN, PID.DOWN,  xa/za, xb/zb, Q, par) * (1./36.)
            + fac_NLO * RaRb_qg * lumi(PID.DOWN, PID.GLUON, xa/za, xb/zb, Q, par) * (1./256.)
            + fac_NLO * RaRb_gq * lumi(PID.GLUON, PID.DOWN, xa/za, xb/zb, Q, par) * (1./256.)
        )
        res += cross_up * (za*zb) * (
                        RaRb_LO * lumi(PID.UP, PID.UP,    xa/za, xb/zb, Q, par) * (1./36.)
            + fac_NLO * RaRb_qq * lumi(PID.UP, PID.UP,    xa/za, xb/zb, Q, par) * (1./36.)
            + fac_NLO * RaRb_qg * lumi(PID.UP, PID.GLUON, xa/za, xb/zb, Q, par) * (1./256.)
            + fac_NLO * RaRb_gq * lumi(PID.GLUON, PID.UP, xa/za, xb/zb, Q, par) * (1./256.)
        )
    if za>xa:
        # RaEb
        res += cross_dn * (za) * (
                        RaEb_LO * lumi(PID.DOWN, PID.DOWN,  xa/za, xb, Q, par) * (1./36.)
            + fac_NLO * RaEb_qq * lumi(PID.DOWN, PID.DOWN,  xa/za, xb, Q, par) * (1./36.)
            + fac_NLO * RaEb_qg * lumi(PID.DOWN, PID.GLUON, xa/za, xb, Q, par) * (1./256.)
            + fac_NLO * RaEb_gq * lumi(PID.GLUON, PID.DOWN, xa/za, xb, Q, par) * (1./256.)
        )
        res += cross_up * (za) * (
                        RaEb_LO * lumi(PID.UP, PID.UP,    xa/za, xb, Q, par) * (1./36.)
            + fac_NLO * RaEb_qq * lumi(PID.UP, PID.UP,    xa/za, xb, Q, par) * (1./36.)
            + fac_NLO * RaEb_qg * lumi(PID.UP, PID.GLUON, xa/za, xb, Q, par) * (1./256.)
            + fac_NLO * RaEb_gq * lumi(PID.GLUON, PID.UP, xa/za, xb, Q, par) * (1./256.)
        )
    if zb>xb:
        # EaRb
        res += cross_dn * (zb) * (
                        EaRb_LO * lumi(PID.DOWN, PID.DOWN,  xa, xb/zb, Q, par) * (1./36.)
            + fac_NLO * EaRb_qq * lumi(PID.DOWN, PID.DOWN,  xa, xb/zb, Q, par) * (1./36.)
            + fac_NLO * EaRb_qg * lumi(PID.DOWN, PID.GLUON, xa, xb/zb, Q, par) * (1./256.)
            + fac_NLO * EaRb_gq * lumi(PID.GLUON, PID.DOWN, xa, xb/zb, Q, par) * (1./256.)
        )
        res += cross_up * (zb) * (
                        EaRb_LO * lumi(PID.UP, PID.UP,    xa, xb/zb, Q, par) * (1./36.)
            + fac_NLO * EaRb_qq * lumi(PID.UP, PID.UP,    xa, xb/zb, Q, par) * (1./36.)
            + fac_NLO * EaRb_qg * lumi(PID.UP, PID.GLUON, xa, xb/zb, Q, par) * (1./256.)
            + fac_NLO * EaRb_gq * lumi(PID.GLUON, PID.UP, xa, xb/zb, Q, par) * (1./256.)
        )
    if True:
        # EaEb
        res += cross_dn * (
                        EaEb_LO * lumi(PID.DOWN, PID.DOWN,  xa, xb, Q, par) * (1./36.)
            + fac_NLO * EaEb_qq * lumi(PID.DOWN, PID.DOWN,  xa, xb, Q, par) * (1./36.)
            + fac_NLO * EaEb_qg * lumi(PID.DOWN, PID.GLUON, xa, xb, Q, par) * (1./256.)
            + fac_NLO * EaEb_gq * lumi(PID.GLUON, PID.DOWN, xa, xb, Q, par) * (1./256.)
        )
        res += cross_up * (
                        EaEb_LO * lumi(PID.UP, PID.UP,    xa, xb, Q, par) * (1./36.)
            + fac_NLO * EaEb_qq * lumi(PID.UP, PID.UP,    xa, xb, Q, par) * (1./36.)
            + fac_NLO * EaEb_qg * lumi(PID.UP, PID.GLUON, xa, xb, Q, par) * (1./256.)
            + fac_NLO * EaEb_gq * lumi(PID.GLUON, PID.UP, xa, xb, Q, par) * (1./256.)
        )
        # print("xa={:e}, xb={:e}, za={:e}, zb={:e} => res={:e} ({:e})".format(xa,xb,za,zb,res,xa*xb))
        # if res != 0. : input()
    return res


def diff_cross_NLO(Ecm: float, za : float, zb: float, Mll: float, Yll: float, par=PARAM) -> float:
    xa = (Mll/Ecm) * math.exp(+Yll)
    xb = (Mll/Ecm) * math.exp(-Yll)
    return par.GeVpb * (2.*Mll/Ecm**2) * (1./(2.*xa*xb*Ecm**2)) * integrand_NLO(xa,xb,za,zb,Mll)


if __name__ == "__main__":
    # Ecm = 8e3
    # tot_cross = scipy.integrate.nquad(lambda M,Y: diff_cross_LO(Ecm,M,Y), [[80.,100.],[-3.6,+3.6]], opts={'epsrel':1e-3})
    # print("total xs = {} pb".format(tot_cross))
    # for z in np.linspace(0.99999, 0.99999999, 20):
    #     RaRb_qg, RaEb_qg, EaRb_qg, EaEb_qg = del_qg(z,0.75)
    #     print("{:e}:  {:e} {:e} {:e} {:e} => {:3}".format(z, RaRb_qg, RaEb_qg, EaRb_qg, EaEb_qg, RaRb_qg+RaEb_qg+EaRb_qg+EaEb_qg))

    # alps = PARAM.pdf.alphasQ(91.18)
    # print("asMZ = {}".format(alps))

    Ecm = 8e3
    # MY_cross_LO = diff_cross_LO(Ecm,100.,0.)
    # print("dXS_LO[M,Y] = {:e} pb".format(MY_cross_LO))
    # MY_cross =  scipy.integrate.nquad(lambda za,zb: diff_cross_NLO(Ecm,za,zb,100.,0.), [[0.,1.],[0.,1.]], opts={'epsrel':1e-3})
    # print("dXS[M,Y]    = {:e} +- {:e} pb".format(*MY_cross))

    # deb_integ = vegas.Integrator([[0.,1.],[0.,1.],[0.,1.],[0.,1.]])
    # deb_integ(lambda x: integrand_NLO(x[0],x[1],x[2],x[3],100.,1.,1.,PARAM), nitn=10, neval=100000)
    # deb_result = deb_integ(lambda x: integrand_NLO(x[0],x[1],x[2],x[3],100.,1.,1.,PARAM), nitn=10, neval=1000000)
    # print(deb_result.summary())
    # raise RuntimeError("die")

    # deb_res = scipy.integrate.nquad(lambda xa,xb,za,zb: integrand_NLO(xa,xb,za,zb,100.,1.,1.,PARAM), [[0.,1.],[0.,1.],[0.,1.],[0.,1.]], opts={'epsrel':1e-3,'epsabs':1e-3})
    # print("deb_res   = {:e} +- {:e} pb".format(*deb_res))


    totXS_LO = scipy.integrate.nquad(lambda M,Y: diff_cross_LO(Ecm,M,Y), [[80.,100.],[-3.6,+3.6]], opts={'epsrel':1e-3})
    print("XS_LO  = {:e} +- {:e} pb".format(*totXS_LO))

    LO_integ = vegas.Integrator([[80.,100.],[-3.6,+3.6]])
    LO_integ(lambda x: diff_cross_LO(Ecm,x[0],x[1]), nitn=10, neval=500)
    LO_result = LO_integ(lambda x: diff_cross_LO(Ecm,x[0],x[1]), nitn=10, neval=1000)
    print(LO_result.summary())

    NLO_integ = vegas.Integrator([[0.,1.],[0.,1.],[80.,100.],[-3.6,+3.6]])
    neval = 1000000
    for i in range(10):
        NLO_result = NLO_integ(lambda x: diff_cross_NLO(Ecm,x[0],x[1],x[2],x[3]), nitn=2, neval=neval)
        print(NLO_result.summary())
        neval *= 2
    # NLO_integ(lambda x: diff_cross_NLO(Ecm,x[0],x[1],x[2],x[3]), nitn=10, neval=1000)
    # NLO_result = NLO_integ(lambda x: diff_cross_NLO(Ecm,x[0],x[1],x[2],x[3]), nitn=10, neval=10000)
    # print(NLO_result.summary())

    # #opts={'epsrel':1e-3,'weight':'alg','wvar':(0.2,0.2)}
    # totXS = scipy.integrate.nquad(lambda za,zb,M,Y: diff_cross_NLO(Ecm,za,zb,M,Y), [[0.,1.],[0.,1.],[80.,100.],[-3.6,+3.6]], opts={'epsrel':1e-3})
    # print("XS_NLO = {:e} +- {:e} pb".format(*totXS))


    # nn = 100
    # for za in np.linspace(0, 1, nn):
    #     print()
    #     for zb in np.linspace(0, 1, nn):
    #         val = integrand_NLO(0.1,0.2,za,zb,100.)
    #         print("{:e} {:e}  {:e}".format(za,zb,val))
