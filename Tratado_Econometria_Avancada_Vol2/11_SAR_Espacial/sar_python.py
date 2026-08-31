# Modelo 11 – SAR (Spatial Autoregressive)
# Autor: Luiz Tiago Wilcke
import numpy as np
from scipy.linalg import inv

np.random.seed(42)
n = 50
# Matriz de pesos espaciais (contigüidade aleatória normalizada)
W = np.random.rand(n, n)
W = (W + W.T)/2
np.fill_diagonal(W, 0)
W = W / W.sum(axis=1, keepdims=True)
rho_v = 0.5
beta_v = np.array([1.0, 2.0])
X = np.column_stack([np.ones(n), np.random.normal(0,1,n)])
u = np.random.normal(0,1,n)
y = inv(np.eye(n) - rho_v*W) @ (X @ beta_v + u)

# Estimação por MQ2E (2SLS espacial)
Wy = W @ y
Z = np.column_stack([X, W @ X[:,1]])  # instrumentos
# 1ª etapa
Wy_hat = Z @ inv(Z.T@Z) @ Z.T @ Wy
# 2ª etapa
X2 = np.column_stack([X, Wy_hat])
theta = inv(X2.T@X2) @ X2.T @ y
print("="*60)
print("MODELO SAR (Spatial Autoregressive)")
print("Autor: Luiz Tiago Wilcke")
print("="*60)
print(f"rho estimado: {theta[-1]:.3f} (verdadeiro={rho_v})")
print(f"beta estimado: {theta[:-1].round(3)} (verdadeiro={beta_v})")
