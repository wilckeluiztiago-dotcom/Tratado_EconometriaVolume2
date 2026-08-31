# ============================================================
# Modelo 04 – DCC-GARCH (Engle, 2002)
# Correlação condicional dinâmica multivariada
# Fonte: Cap. 19, pp. 181–183 | DCC: p. 182
# Autor: Luiz Tiago Wilcke
# ============================================================

import numpy as np
from scipy.optimize import minimize

np.random.seed(42)
T, N = 800, 2

# --- Simular retornos com correlação variante no tempo ---
rho_t = 0.3 + 0.4 * np.sin(np.linspace(0, 6*np.pi, T))
retornos = np.zeros((T, N))
h = np.ones(N)
for t in range(T):
    corr = np.array([[1, rho_t[t]], [rho_t[t], 1]])
    D = np.diag(np.sqrt(h))
    H = D @ corr @ D
    retornos[t] = np.random.multivariate_normal(np.zeros(N), H)
    # GARCH(1,1) univariado simplificado
    h = 0.01 + 0.05 * retornos[t]**2 + 0.90 * h

# --- Etapa 1: GARCH univariado ---
def garch_loglik(theta, r):
    omega, alpha, beta = theta
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e10
    T = len(r)
    h = np.empty(T); h[0] = np.var(r)
    ll = 0.0
    for t in range(1, T):
        h[t] = omega + alpha * r[t-1]**2 + beta * h[t-1]
        ll += -0.5 * (np.log(2*np.pi) + np.log(h[t]) + r[t]**2 / h[t])
    return -ll

params_u = []
resid_std = np.zeros_like(retornos)
for i in range(N):
    res = minimize(garch_loglik, [0.01, 0.05, 0.9], args=(retornos[:, i],),
                   method="L-BFGS-B", bounds=[(1e-6, None), (0, 0.3), (0, 0.99)])
    params_u.append(res.x)
    omega, alpha, beta = res.x
    h = np.empty(T); h[0] = np.var(retornos[:, i])
    for t in range(1, T):
        h[t] = omega + alpha * retornos[t-1, i]**2 + beta * h[t-1]
    resid_std[:, i] = retornos[:, i] / np.sqrt(h)

# --- Etapa 2: DCC ---
def dcc_loglik(phi, e):
    a, b = phi
    if a < 0 or b < 0 or a + b >= 0.999:
        return 1e10
    T, N = e.shape
    Qbar = np.corrcoef(e.T)
    Q = Qbar.copy()
    ll = 0.0
    for t in range(T):
        if t > 0:
            Q = (1 - a - b) * Qbar + a * np.outer(e[t-1], e[t-1]) + b * Q
        R = Q / np.outer(np.sqrt(np.diag(Q)), np.sqrt(np.diag(Q)))
        try:
            ll += -0.5 * (np.log(np.linalg.det(R)) + e[t] @ np.linalg.inv(R) @ e[t]
                          - e[t] @ e[t])
        except:
            return 1e10
    return -ll

res_dcc = minimize(dcc_loglik, [0.05, 0.9], args=(resid_std,),
                   method="L-BFGS-B", bounds=[(0, 0.3), (0, 0.99)])
print("=" * 70)
print("DCC-GARCH (Engle 2002)")
print("Fonte: Cap. 19, pp. 181–183 | Especificação DCC: p. 182")
print("Autor: Luiz Tiago Wilcke")
print("=" * 70)
print(f"GARCH univariado (omega, alpha, beta):")
for i, p in enumerate(params_u):
    print(f"  Série {i+1}: {p.round(4)}")
print(f"DCC (a, b): {res_dcc.x.round(4)}")
print(f"Correlação incondicional amostral: {np.corrcoef(retornos.T)[0,1]:.3f}")
