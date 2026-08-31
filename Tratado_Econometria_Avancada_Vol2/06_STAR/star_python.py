# Modelo 06 – STAR (Smooth Transition Autoregressive)
# Autor: Luiz Tiago Wilcke
import numpy as np
from scipy.optimize import minimize

np.random.seed(42)
n = 300
gamma_v, c_v = 2.0, 0.0
y = np.zeros(n)
for t in range(1, n):
    G = 1 / (1 + np.exp(-gamma_v * (y[t-1] - c_v)))
    y[t] = (0.4 + 0.5*y[t-1])*(1-G) + (-0.1 + 0.2*y[t-1])*G + np.random.normal(0, 0.7)

def negloglik(theta):
    gamma, c, a0, a1, b0, b1 = theta
    if gamma <= 0: return 1e10
    e = np.zeros(n-1)
    for t in range(1, n):
        G = 1/(1+np.exp(-gamma*(y[t-1]-c)))
        e[t-1] = y[t] - ((a0+a1*y[t-1])*(1-G) + (b0+b1*y[t-1])*G)
    return 0.5*np.sum(e**2)

res = minimize(negloglik, [1.5, 0, 0.3, 0.4, 0, 0.1], method="Nelder-Mead")
print("="*60)
print("MODELO STAR (LSTAR)\nAutor: Luiz Tiago Wilcke")
print("="*60)
print("Parâmetros estimados (gamma, c, a0, a1, b0, b1):")
print(np.round(res.x, 4))
print("Verdadeiros approx: gamma=2, c=0, a0=0.4, a1=0.5, b0=-0.1, b1=0.2")
