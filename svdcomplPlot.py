from gc import DEBUG_COLLECTABLE
import torch
import matplotlib.pyplot as plt
import numpy as np


def complex(R, L, x, adimensional = True, debug = False):
    alpha = ((1/3)*L - (1/2))*(4/R)
    beta = (1/(R**2))
    gamma = (2*L)/(3*R**3)
    delta = - x**4 + alpha * x**3 - beta * x**2 + gamma * x
    delta = delta * (R**4)/4 * (not adimensional) + adimensional * delta
    if debug:
        print(alpha, beta, gamma)
    return delta

R,L = 10, 1000

x = np.linspace(0, 1, 100)

delta = [complex(R,L,i, False) for i in x]
delta_ad = [complex(R,L,i) for i in x]

plt.plot(delta, label="original")
# plt.plot(delta_ad, label="adimensionalized")
plt.legend()
plt.show()

