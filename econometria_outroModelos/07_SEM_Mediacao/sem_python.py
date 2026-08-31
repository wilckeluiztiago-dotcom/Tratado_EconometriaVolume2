# ============================================================
# Modelo 07 – SEM e Mediação Causal
# Fonte: Cap. 21, pp. 202–211 | Mediação: pp. 204–205
# Autor: Luiz Tiago Wilcke
# ============================================================

import numpy as np
from numpy.linalg import inv

np.random.seed(42)
n = 800

# Modelo estrutural verdadeiro:
# M = a*X + e_m
# Y = c'*X + b*M + e_y
# Efeito indireto = a*b ; direto = c' ; total = c' + a*b
a_true, b_true, c_true = 0.6, 0.5, 0.3
X = np.random.normal(0, 1, n)
M = a_true * X + np.random.normal(0, 1, n)
Y = c_true * X + b_true * M + np.random.normal(0, 1, n)

# --- Estimação por equações (Baron-Kenny / produto de coeficientes) ---
# Regressão M ~ X
a_hat = np.cov(X, M)[0, 1] / np.var(X)
# Regressão Y ~ X + M
Z = np.column_stack([np.ones(n), X, M])
coef = inv(Z.T @ Z) @ Z.T @ Y
c_prime, b_hat = coef[1], coef[2]
indireto = a_hat * b_hat
total = c_prime + indireto

# Bootstrap para IC do efeito indireto
B = 500
ind_boot = []
for _ in range(B):
    idx = np.random.choice(n, n, replace=True)
    Xb, Mb, Yb = X[idx], M[idx], Y[idx]
    a_b = np.cov(Xb, Mb)[0, 1] / np.var(Xb)
    Zb = np.column_stack([np.ones(n), Xb, Mb])
    cb = inv(Zb.T @ Zb) @ Zb.T @ Yb
    ind_boot.append(a_b * cb[2])
ind_boot = np.array(ind_boot)
ic = np.percentile(ind_boot, [2.5, 97.5])

print("=" * 70)
print("SEM / MEDIAÇÃO CAUSAL (produto de coeficientes + bootstrap)")
print("Fonte: Cap. 21, pp. 202–211 | Mediação contemporânea: pp. 204–205")
print("Autor: Luiz Tiago Wilcke")
print("=" * 70)
print(f"a (X->M): {a_hat:.3f} (verdadeiro={a_true})")
print(f"b (M->Y): {b_hat:.3f} (verdadeiro={b_true})")
print(f"c' (direto): {c_prime:.3f} (verdadeiro={c_true})")
print(f"Efeito indireto a*b: {indireto:.3f} (verdadeiro={a_true*b_true})")
print(f"Efeito total: {total:.3f}")
print(f"IC 95% bootstrap indireto: [{ic[0]:.3f}, {ic[1]:.3f}]")
