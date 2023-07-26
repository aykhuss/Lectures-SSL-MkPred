#!/usr/bin/env python

import math
import random
import sys

random.seed(42)
alphas = 0.118

def generate_event(Q2_start: float, Q2_cutoff: float, CX: float):
  sudakov = 1.  # initialize Sudakov to the starting scale
  fac = alphas*CX/(2.*math.pi)
  Qlist = []
  while True:
    r = random.uniform(0.,1.)
    sudakov *= r
    #> sudakov = exp( -[alphas*CX/(2.*pi)] * log^2[Q2/Q2_start] )
    #> determine Q2 from the associated sudakov
    L2 = - math.log(sudakov) / fac
    Q2 = Q2_start * math.exp(-math.sqrt(L2))
    if Q2 < Q2_cutoff:
      break
    Qlist.append( math.sqrt(Q2) )
  if len(Qlist) > 1:
    print("#summary2 {} {} {} {}".format(len(Qlist),sum(Qlist),Qlist[0],Qlist))

if __name__ == "__main__":
  if len(sys.argv) < 3:
    raise RuntimeError("I expect at least two arguments:  Q_start [g|q]")
  Q_start = float(sys.argv[1])  # the hard scale
  Q_cutoff = 1  # shower cutoff (PS stops -> hand over to hadronization)
  if sys.argv[2] == "q":
    CX = 4./3.  # quark
  elif sys.argv[2] == "g":
    CX = 3.     # gluon
  else:
    raise RuntimeError("unrecognised parton: {}".format(sys.argv[2]))
  if len(sys.argv) >= 4:
    alphas = float(sys.argv[3])
  if len(sys.argv) >= 5:
    nevents = int(sys.argv[4])
  else:
    nevents = 1000
  for i in range(nevents):
    print("# event {} [{} {} {} {} {}]".format(i,Q_start,sys.argv[2],CX,alphas,nevents))
    generate_event(Q_start**2, Q_cutoff**2, CX)
