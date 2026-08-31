# ============================================================
# Modelo 04 – GMM (Método Generalizado dos Momentos)
# Tratado de Econometria Avançada – Volume II
# Autor: Luiz Tiago Wilcke
# ============================================================

import numpy as np
import pandas as pd
from scipy.optimize import minimize

np.random.seed(42)
n = 500

# Modelo estrutural: y = beta0 + beta1 * x + e
# x é endógeno; z1, z2 são instrumentos
x_latente = np.random.normal(0, 1, n)
e = 0.7 * x_latente + np.random.normal(0, 1, n)   # endogeneidade
x = x_latente + np.random.normal(0, 0.5, n)
z1 = 0.8 * x_latente + np.random.normal(0, 1, n)
z2 = 0.5 * x_latente + np.random.normal(0, 1, n)
beta_verdadeiro = np.array([1.0, 2.0])
y = beta_verdadeiro[0] + beta_verdadeiro[1] * x + e

dados = pd.DataFrame({
    "resultado": y, "explicativa": x,
    "instrumento1": z1, "instrumento2": z2
})

def momentos(beta, y, x, z):
    resid = y - beta[0] - beta[1] * x
    return np.column_stack([resid, resid * z[:, 0], resid * z[:, 1]])

def gmm_objetivo(beta, y, x, z, W):
    g = momentos(beta, y, x, z)
    g_bar = g.mean(axis=0)
    return n * g_bar @ W @ g_bar

# Primeira etapa (W = I)
W1 = np.eye(3)
res1 = minimize(lambda b: gmm_objetivo(b, y, x, np.column_stack([z1, z2]), W1),
                x0=[0, 0], method="BFGS")
beta1 = res1.x

# Matriz de ponderação ótima (2S)
g1 = momentos(beta1, y, x, np.column_stack([z1, z2]))
S = np.cov(g1.T)
W2 = np.linalg.inv(S)
res2 = minimize(lambda b: gmm_objetivo(b, y, x, np.column_stack([z1, z2]), W2),
                x0=beta1, method="BFGS")
beta_gmm = res2.x

print("=" * 60)
print("GMM – MÉTODO GENERALIZADO DOS MOMENTOS (2 etapas)")
print("Autor: Luiz Tiago Wilcke")
print("=" * 60)
print(f"Beta GMM estimado: intercepto = {beta_gmm[0]:.4f}, slope = {beta_gmm[1]:.4f}")
print(f"Beta verdadeiro:   intercepto = {beta_verdadeiro[0]:.1f}, slope = {beta_verdadeiro[1]:.1f}")
print(f"Valor da função objetivo (J-stat): {res2.fun:.4f}")
