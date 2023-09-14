#!/usr/bin/env python

import math
import random
import sys
import numpy as np

random.seed(42)
alphas = 0.118


def generate_emissions(kt_max: float, kt_cut: float, CX: float) -> list[float]:
    emissions = list()
    fac = CX * alphas / math.pi  # save common factor
    sudakov = 1.  # initialize to the starting scale
    while True:
        sudakov *= random.uniform(0., 1.)
        #> invert `r = sudakov(kt, kt_new)`
        L2 = -math.log(sudakov) / fac
        kt = kt_max * math.exp(-math.sqrt(L2))
        if kt <= kt_cut:
            break
        emissions.append(kt)
    return emissions


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise RuntimeError("I expect at least two arguments:  kt_max [g|q]")
    kt_max = float(sys.argv[1])  # the hard scale
    kt_cut = 1.  # shower cutoff
    if sys.argv[2].lower() == "q":
        CX = 4. / 3.
    elif sys.argv[2].lower() == "g":
        CX = 3.
    else:
        raise RuntimeError("unrecognised parton: {}".format(sys.argv[2]))
    if len(sys.argv) >= 4:
        alphas = float(sys.argv[3])
    if len(sys.argv) >= 5:
        nevents = int(sys.argv[4])
    else:
        nevents = 1000
    for i in range(nevents):
        print("#event {} [{} {} {} {} {}]".format(i, kt_max, sys.argv[2], CX,
                                                  alphas, nevents))
        emissions = generate_emissions(kt_max, kt_cut, CX)
        if len(emissions) > 0:
            print("#summary {} {} {} {} {}".format(
                len(emissions), sum(emissions),
                math.log(sum(emissions) / kt_max), emissions[0],
                math.log(emissions[0] / kt_max)))
