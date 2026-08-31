# ============================================================
# Modelo 16 – Modelo de Riscos Proporcionais de Cox
# Fonte: Cap. 26, pp. 248–256 | Cox: pp. 250–251
# Autor: Luiz Tiago Wilcke
# ============================================================

import numpy as np
from scipy.optimize import minimize

np.random.seed(42)
n = 500
X = np.random.normal(0, 1, (n, 2))
beta_true = np.array([0.8, -0.5])
# Tempo de falha (Weibull com riscos proporcionais)
lam = np.exp(X @ beta_true)
tempo = np.random.weibull(1.5, n) / lam
censor = np.random.uniform(0, tempo.max()*0.8, n)
evento = (tempo <= censor).astype(float)
tempo_obs = np.minimum(tempo, censor)

# Ordenar por tempo
ord_ = np.argsort(tempo_obs)
t = tempo_obs[ord_]
e = evento[ord_]
Xo = X[ord_]

def cox_negll(beta, X, e):
    """Verossimilhança parcial de Cox."""
    xb = X @ beta
    ll = 0.0
    for i in range(len(e)):
        if e[i] == 0:
            continue
        # risco do conjunto sob risco R(t_i) = i..n-1 (já ordenado)
        risk = xb[i:]
        m = risk.max()
        ll += xb[i] - (m + np.log(np.exp(risk - m).sum()))
    return -ll

res = minimize(cox_negll, [0.0, 0.0], args=(Xo, e), method="BFGS")
print("=" * 70)
print("COX – RISCOS PROPORCIONAIS (verossimilhança parcial)")
print("Fonte: Cap. 26, pp. 248–256 | Cox (1972): pp. 250–251")
print("Autor: Luiz Tiago Wilcke")
print("=" * 70)
print(f"Beta Cox: {res.x.round(3)}")
print(f"Beta verdadeiro: {beta_true}")
print(f"Taxa de censura: {(1-e.mean()):.3f}")
print(f"Hazard ratio X1: {np.exp(res.x[0]):.3f}")
