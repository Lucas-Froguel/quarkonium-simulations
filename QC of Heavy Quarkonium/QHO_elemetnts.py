import numpy as np
import matplotlib.pyplot as plt
import json
import scipy.integrate as integrate
import scipy.special as sp
from scipy.constants import hbar
import mpmath

omega = 562.9  # MeV
#  mass = 1.0
kappa = 0.4063  # MeV
sigma = 441.6 ** 2  # MeV
mu = 637.5 # MeV
b = 1 / np.sqrt(mu * omega)

def kdelta(n, m):
    return 1 if n==m else 0

def T(n, m, w: float = 0):
    t1 = (2*n + 3/2) * kdelta(n, m)
    t2 = - np.sqrt(n*(n+1/2)) * kdelta(n, m+1)
    t3 = - np.sqrt((n+1)*(n+3/2)) * kdelta(n, m-1)
    return (w/2) * (t1 + t2 +t3)

def V(n, m, b: float = 0, k: float = 0, o: float = 0):
    sqrt_gammas = np.sqrt( (sp.gamma(m+3/2) * sp.gamma(n+3/2)) / (sp.gamma(m+1) * sp.gamma(n+1)) ) 
    
    v1 = -k * (-1)**(n+m) * (4 / (b * np.pi * (1+2*n))) \
    * sqrt_gammas \
    * float(mpmath.hyp3f2(1/2, 1, -m, 3/2, 1/2 - n, 1))

    v2 = o * (-1)**(n+m) * ((4 * b) / (np.pi * (1 - 4 * n**2))) \
    * sqrt_gammas \
    * float(mpmath.hyp2f1(2, -m, 3/2 - n, 1))

    return v1 + v2

N = 3
Vnm = {}
for n in range(N):
    for m in range(N):
        Vnm[f"{n}{m}"] = V(n, m, b=b, k=kappa, o=sigma)

Tk = {
    "0": 21 * omega / 4,
    "1": 3 * omega / 4,
    "2": 7 * omega / 4,
    "3": 11 * omega / 4,
    "4": -np.sqrt(3/2),
    "5": -np.sqrt(5),
    "6": -np.sqrt(3/2),
    "7": -np.sqrt(5),
    "8": 0,
    "9": 0
}


H = {f"{k}": {} for k in range(N**2+1)}

H["0"]["coeff"] = (1/2) * (Tk["0"] + Vnm["00"] + Vnm["11"] + Vnm["22"])
H["0"]["operator"] = 'III'

H["1"]["coeff"] = -(1/2) * (Tk["1"] + Vnm["00"])
H["1"]["operator"] = 'ZII'

H["2"]["coeff"] = -(1/2) * (Tk["2"] + Vnm["11"])
H["2"]["operator"] = 'IZI'

H["3"]["coeff"] = -(1/2) * (Tk["3"] + Vnm["22"])
H["3"]["operator"] = 'IIZ'

H["4"]["coeff"] = (1/4) * (Tk["4"] + 2*Vnm["01"])
H["4"]["operator"] = 'XXI'

H["5"]["coeff"] = (1/4) * (Tk["5"] + 2*Vnm["12"])
H["5"]["operator"] = 'IXX'

H["6"]["coeff"] = (1/4) * (Tk["6"] + 2*Vnm["01"])
H["6"]["operator"] = 'YYI'

H["7"]["coeff"] = (1/4) * (Tk["7"] + 2*Vnm["12"])
H["7"]["operator"] = 'IYY'

H["8"]["coeff"] = (1/2) * (Tk["8"] + Vnm["02"])
H["8"]["operator"] = 'XZX'

H["9"]["coeff"] = (1/2) * (Tk["9"] + Vnm["00"])
H["9"]["operator"] = 'YZY'


with open("H_data.json", "w+") as file:
    json.dump(H, file, indent=2)
