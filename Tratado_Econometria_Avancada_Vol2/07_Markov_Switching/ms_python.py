# Modelo 07 – Markov Switching
# Autor: Luiz Tiago Wilcke
import numpy as np
from scipy.stats import norm

np.random.seed(42)
n = 250
# 2 regimes: média baixa e alta
P = np.array([[0.9, 0.1], [0.2, 0.8]])  # matriz de transição
mu = [-1.0, 2.0]
sig = [0.8, 1.2]
estado = 0
y = np.zeros(n)
estados_v = np.zeros(n, dtype=int)
for t in range(n):
    estados_v[t] = estado
    y[t] = mu[estado] + np.random.normal(0, sig[estado])
    estado = np.random.choice([0,1], p=P[estado])

# Filtro de Hamilton simplificado (estimação por EM aproximada)
def hamilton_filter(y, mu, sig, P):
    n = len(y)
    xi = np.zeros((n, 2))
    xi[0] = [0.5, 0.5]
    lik = 0.0
    for t in range(1, n):
        pred = xi[t-1] @ P
        dens = np.array([norm.pdf(y[t], mu[0], sig[0]), norm.pdf(y[t], mu[1], sig[1])])
        xi[t] = pred * dens
        xi[t] /= xi[t].sum()
        lik += np.log((pred * dens).sum())
    return xi, lik

xi, loglik = hamilton_filter(y, mu, sig, P)
print("="*60)
print("MARKOV SWITCHING – Filtro de Hamilton")
print("Autor: Luiz Tiago Wilcke")
print("="*60)
print(f"Log-verossimilhança: {loglik:.2f}")
print(f"Prob. média regime 1 (estimado): {xi[:,0].mean():.3f}")
print(f"Proporção verdadeira regime 0: {(estados_v==0).mean():.3f}")
