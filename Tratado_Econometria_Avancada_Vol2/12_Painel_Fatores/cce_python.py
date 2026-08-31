# Modelo 12 – CCE (Common Correlated Effects) de Pesaran
# Autor: Luiz Tiago Wilcke
import numpy as np

np.random.seed(42)
N, T = 30, 40
# Fatores comuns
f = np.random.normal(0, 1, (T, 2))
lambda_i = np.random.normal(0, 1, (N, 2))
beta_v = 1.5
x = np.random.normal(0, 1, (N, T))
y = np.zeros((N, T))
for i in range(N):
    y[i] = beta_v * x[i] + lambda_i[i] @ f.T + np.random.normal(0, 1, T)

# CCE: médias transversais como proxies dos fatores
y_bar = y.mean(axis=0)
x_bar = x.mean(axis=0)
betas = []
for i in range(N):
    # Regressão: y_it = b x_it + c1 y_bar + c2 x_bar + e
    Z = np.column_stack([x[i], y_bar, x_bar])
    b = np.linalg.lstsq(Z, y[i], rcond=None)[0]
    betas.append(b[0])
beta_cce = np.mean(betas)
print("="*60)
print("CCE – Common Correlated Effects (Pesaran 2006)")
print("Autor: Luiz Tiago Wilcke")
print("="*60)
print(f"Beta CCE médio: {beta_cce:.3f} (verdadeiro={beta_v})")
