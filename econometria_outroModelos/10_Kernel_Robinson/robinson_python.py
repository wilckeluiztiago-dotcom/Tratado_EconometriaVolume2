# ============================================================
# Modelo 10 – Estimador Parcialmente Linear de Robinson (1988)
# Fonte: Cap. 29, pp. 277–285 | Robinson: pp. 279–280
# Autor: Luiz Tiago Wilcke
# ============================================================

import numpy as np
from numpy.linalg import inv

np.random.seed(42)
n = 600
# Y = X beta + g(Z) + e
beta_true = np.array([1.0, -0.5])
X = np.random.normal(0, 1, (n, 2))
Z = np.random.uniform(-2, 2, n)
g_true = np.sin(Z) + 0.5 * Z**2
Y = X @ beta_true + g_true + np.random.normal(0, 0.5, n)

def kernel_nw(z0, Z, Y, h):
    """Nadaraya-Watson."""
    u = (Z - z0) / h
    K = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
    w = K / (K.sum() + 1e-12)
    return w @ Y

# Bandwidth
h = 1.06 * np.std(Z) * n**(-1/5)

# Residualização: Y - E[Y|Z], X - E[X|Z]
Y_tilde = np.array([Y[i] - kernel_nw(Z[i], Z, Y, h) for i in range(n)])
X_tilde = np.zeros_like(X)
for j in range(X.shape[1]):
    X_tilde[:, j] = [X[i, j] - kernel_nw(Z[i], Z, X[:, j], h) for i in range(n)]

beta_robinson = inv(X_tilde.T @ X_tilde) @ (X_tilde.T @ Y_tilde)

# g estimado
resid = Y - X @ beta_robinson
g_hat = np.array([kernel_nw(z, Z, resid, h) for z in np.linspace(-2, 2, 50)])

print("=" * 70)
print("ROBINSON (1988) – REGRESSÃO PARCIALMENTE LINEAR")
print("Fonte: Cap. 29, pp. 277–285 | Procedimento de Robinson: pp. 279–280")
print("Autor: Luiz Tiago Wilcke")
print("=" * 70)
print(f"Beta Robinson: {beta_robinson.round(3)}")
print(f"Beta verdadeiro: {beta_true}")
print(f"Bandwidth (Silverman): {h:.3f}")
