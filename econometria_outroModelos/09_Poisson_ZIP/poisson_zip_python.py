# ============================================================
# Modelo 09 – Poisson, Binomial Negativa e ZIP
# Fonte: Cap. 28, pp. 267–275 | ZIP: pp. 269–270
# Autor: Luiz Tiago Wilcke
# ============================================================

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

np.random.seed(42)
n = 800
X = np.column_stack([np.ones(n), np.random.normal(0, 1, n), np.random.binomial(1, 0.4, n)])
beta_true = np.array([0.5, 0.4, -0.3])
mu = np.exp(X @ beta_true)
# ZIP: com prob pi, y=0 estrutural; senão Poisson(mu)
pi_true = 0.25
y = np.zeros(n, dtype=int)
for i in range(n):
    if np.random.rand() < pi_true:
        y[i] = 0
    else:
        y[i] = np.random.poisson(mu[i])

def poisson_negll(beta, y, X):
    xb = X @ beta
    return -np.sum(y * xb - np.exp(xb) - gammaln(y + 1))

def zip_negll(theta, y, X):
    beta, logit_pi = theta[:-1], theta[-1]
    pi = 1 / (1 + np.exp(-logit_pi))
    mu = np.exp(X @ beta)
    ll = 0.0
    for i in range(len(y)):
        if y[i] == 0:
            ll += np.log(pi + (1 - pi) * np.exp(-mu[i]) + 1e-12)
        else:
            ll += np.log(1 - pi + 1e-12) + y[i]*np.log(mu[i]+1e-12) - mu[i] - gammaln(y[i]+1)
    return -ll

res_p = minimize(poisson_negll, np.zeros(3), args=(y, X), method="BFGS")
res_z = minimize(zip_negll, np.zeros(4), args=(y, X), method="BFGS")
pi_hat = 1 / (1 + np.exp(-res_z.x[-1]))

print("=" * 70)
print("POISSON / ZIP (Zero-Inflated Poisson)")
print("Fonte: Cap. 28, pp. 267–275 | ZIP: pp. 269–270")
print("Autor: Luiz Tiago Wilcke")
print("=" * 70)
print(f"Poisson beta: {res_p.x.round(3)}")
print(f"ZIP beta:     {res_z.x[:-1].round(3)} (verdadeiro={beta_true})")
print(f"ZIP pi:       {pi_hat:.3f} (verdadeiro={pi_true})")
print(f"Proporção de zeros: {(y==0).mean():.3f}")
print(f"Média de y: {y.mean():.3f}")
