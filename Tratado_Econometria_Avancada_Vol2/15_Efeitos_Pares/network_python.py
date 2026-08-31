# Modelo 15 – Efeitos de Pares (Network / Linear-in-means)
# Autor: Luiz Tiago Wilcke
import numpy as np

np.random.seed(42)
n = 100
# Rede aleatória (grafo Erdős–Rényi)
A = (np.random.rand(n,n) < 0.1).astype(float)
np.fill_diagonal(A, 0)
A = (A + A.T)/2  # não-dirigida
# Normalização por linha (matriz de médias dos pares)
grau = A.sum(axis=1, keepdims=True)
grau[grau==0] = 1
G = A / grau
# Modelo: y = alpha + beta * G y + gamma * x + delta * G x + e
beta_v, gamma_v, delta_v = 0.4, 1.0, 0.3
x = np.random.normal(0,1,n)
e = np.random.normal(0,1,n)
# Forma reduzida
I_minus = np.eye(n) - beta_v * G
y = np.linalg.inv(I_minus) @ (gamma_v * x + delta_v * G @ x + e)

# Estimação 2SLS (instrumentos: G^2 x etc.)
Gx = G @ x
G2x = G @ Gx
Z = np.column_stack([np.ones(n), x, Gx, G2x])  # instrumentos
Gy = G @ y
# 1ª etapa para Gy
Gy_hat = Z @ np.linalg.lstsq(Z, Gy, rcond=None)[0]
X2sls = np.column_stack([np.ones(n), Gy_hat, x, Gx])
theta = np.linalg.lstsq(X2sls, y, rcond=None)[0]
print("="*60)
print("EFEITOS DE PARES (Linear-in-means / Bramoullé et al.)")
print("Autor: Luiz Tiago Wilcke")
print("="*60)
print(f"beta (endógeno) estimado: {theta[1]:.3f} (verdadeiro={beta_v})")
print(f"gamma estimado: {theta[2]:.3f} (verdadeiro={gamma_v})")
print(f"delta estimado: {theta[3]:.3f} (verdadeiro={delta_v})")
