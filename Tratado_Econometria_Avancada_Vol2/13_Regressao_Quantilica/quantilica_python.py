# Modelo 13 – Regressão Quantílica
# Autor: Luiz Tiago Wilcke
import numpy as np
from scipy.optimize import minimize

np.random.seed(42)
n = 500
x = np.random.uniform(0, 10, n)
# Heterocedasticidade: efeito no quantil 0.5 = 1, no 0.9 maior
y = 2 + 1*x + (0.5 + 0.3*x)*np.random.normal(0,1,n)

def quantil_loss(beta, y, x, tau):
    resid = y - beta[0] - beta[1]*x
    return np.sum(np.where(resid >= 0, tau*resid, (tau-1)*resid))

for tau in [0.25, 0.5, 0.75, 0.9]:
    res = minimize(lambda b: quantil_loss(b, y, x, tau), [0,0], method="Nelder-Mead")
    print(f"Tau={tau:.2f} | intercepto={res.x[0]:.3f} | slope={res.x[1]:.3f}")
print("="*60)
print("REGRESSÃO QUANTÍLICA")
print("Autor: Luiz Tiago Wilcke")
print("="*60)
