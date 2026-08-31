# ============================================================
# Modelo 17 – Equações Simultâneas: 2SLS, LIML
# Fonte: Cap. 23, pp. 222–230 | 2SLS: p. 224 | LIML: p. 225
# Autor: Luiz Tiago Wilcke
# ============================================================

import numpy as np
from numpy.linalg import inv, eig

np.random.seed(42)
n = 500
# Forma estrutural:
# y1 = beta12 * y2 + gamma11 * z1 + e1
# y2 = beta21 * y1 + gamma22 * z2 + e2
beta12_true, beta21_true = 0.6, 0.4
gamma11, gamma22 = 1.0, 1.2
z1 = np.random.normal(0, 1, n)
z2 = np.random.normal(0, 1, n)
# Forma reduzida (simular via erros correlacionados)
e1 = np.random.normal(0, 1, n)
e2 = 0.5*e1 + np.random.normal(0, 1, n)
# Resolver sistema
# y1 - b12 y2 = g11 z1 + e1
# -b21 y1 + y2 = g22 z2 + e2
A = np.array([[1, -beta12_true], [-beta21_true, 1]])
Y = np.zeros((n, 2))
for i in range(n):
    rhs = np.array([gamma11*z1[i] + e1[i], gamma22*z2[i] + e2[i]])
    Y[i] = inv(A) @ rhs
y1, y2 = Y[:, 0], Y[:, 1]

# 2SLS para equação 1: y1 = b12 y2 + g11 z1 + e1
# Instrumentos: z1, z2
Z = np.column_stack([np.ones(n), z1, z2])
# 1ª etapa: y2 ~ Z
P = Z @ inv(Z.T @ Z) @ Z.T
y2_hat = P @ y2
# 2ª etapa
X2 = np.column_stack([y2_hat, z1])
b_2sls = inv(X2.T @ X2) @ X2.T @ y1

# OLS viesado
X_ols = np.column_stack([y2, z1])
b_ols = inv(X_ols.T @ X_ols) @ X_ols.T @ y1

print("=" * 70)
print("EQUAÇÕES SIMULTÂNEAS – 2SLS")
print("Fonte: Cap. 23, pp. 222–230 | 2SLS: p. 224")
print("Autor: Luiz Tiago Wilcke")
print("=" * 70)
print(f"OLS  beta12 (viesado): {b_ols[0]:.3f}")
print(f"2SLS beta12:           {b_2sls[0]:.3f} (verdadeiro={beta12_true})")
print(f"2SLS gamma11:          {b_2sls[1]:.3f} (verdadeiro={gamma11})")
