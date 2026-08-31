# ============================================================
# Modelo 08 – Tobit e Seleção de Heckman (Heckit)
# Fonte: Cap. 27, pp. 257–265 | Tobit: pp. 257–258 | Heckman: pp. 258–259
# Autor: Luiz Tiago Wilcke
# ============================================================

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from numpy.linalg import inv

np.random.seed(42)
n = 1000

# --- Tobit (censura em zero) ---
beta_t = np.array([1.0, 1.5])
X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
ystar = X @ beta_t + np.random.normal(0, 1, n)
y_tobit = np.maximum(ystar, 0)

def tobit_negll(theta, y, X):
    beta, log_sig = theta[:-1], theta[-1]
    sig = np.exp(log_sig)
    xb = X @ beta
    ll = 0.0
    for i in range(len(y)):
        if y[i] > 0:
            ll += -0.5*np.log(2*np.pi) - np.log(sig) - 0.5*((y[i]-xb[i])/sig)**2
        else:
            ll += np.log(norm.cdf(-xb[i]/sig) + 1e-12)
    return -ll

res_t = minimize(tobit_negll, [0.5, 0.5, 0.0], args=(y_tobit, X), method="BFGS")
print("=" * 70)
print("TOBIT E HECKMAN (HECKIT)")
print("Fonte: Cap. 27, pp. 257–265")
print("Autor: Luiz Tiago Wilcke")
print("=" * 70)
print(f"Tobit beta: {res_t.x[:-1].round(3)} (verdadeiro={beta_t})")
print(f"Tobit sigma: {np.exp(res_t.x[-1]):.3f}")

# --- Heckman seleção ---
# Seleção: D* = Z gamma + u ; D = 1{D*>0}
# Resultado: Y = X beta + e ; observado se D=1
# corr(u,e) = rho
gamma_true = np.array([0.5, 1.0])
beta_h = np.array([1.0, 0.8])
rho_true, sig_e = 0.6, 1.0
Z = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
X2 = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
u = np.random.normal(0, 1, n)
e = rho_true * u + np.sqrt(1-rho_true**2) * np.random.normal(0, 1, n) * sig_e
D = (Z @ gamma_true + u > 0).astype(float)
Y = X2 @ beta_h + e
Y_obs = np.where(D == 1, Y, np.nan)

# Heckit 2 etapas
# 1) Probit
def probit_negll(g, D, Z):
    xb = Z @ g
    p = norm.cdf(xb)
    p = np.clip(p, 1e-12, 1-1e-12)
    return -np.sum(D*np.log(p) + (1-D)*np.log(1-p))

res_p = minimize(probit_negll, [0.0, 0.0], args=(D, Z), method="BFGS")
gamma_hat = res_p.x
xb = Z @ gamma_hat
mills = norm.pdf(xb) / (norm.cdf(xb) + 1e-12)  # inverso da razão de Mills

# 2) MQO com Mills
mask = D == 1
Xm = np.column_stack([X2[mask], mills[mask]])
beta_heck = inv(Xm.T @ Xm) @ Xm.T @ Y[mask]
print(f"\nHeckman beta (X): {beta_heck[:2].round(3)} (verdadeiro={beta_h})")
print(f"Coef. Mills (lambda): {beta_heck[2]:.3f}")
print(f"Taxa de seleção: {D.mean():.3f}")
