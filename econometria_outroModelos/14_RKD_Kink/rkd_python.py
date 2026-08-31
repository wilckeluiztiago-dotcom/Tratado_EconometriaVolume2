# ============================================================
# Modelo 14 – Regression Kink Design (RKD)
# Fonte: Cap. 34, pp. 326–333
# Autor: Luiz Tiago Wilcke
# ============================================================

import numpy as np
from numpy.linalg import lstsq

np.random.seed(42)
n = 2000
# Running variable
x = np.random.uniform(-1, 1, n)
c = 0.0
# Tratamento com kink: b(x) = 0.5*x se x<c; 0.5*x + 1.0*(x-c) se x>=c
# (mudança de slope = 1.0)
b = np.where(x < c, 0.5 * x, 0.5 * x + 1.0 * (x - c))
# Resultado: Y = tau * b(x) + f(x) + e
tau_true = 2.0
f = 1.0 + 0.3 * x - 0.1 * x**2
Y = tau_true * b + f + np.random.normal(0, 0.5, n)

# Estimação local linear dos slopes à esquerda e direita
h = 0.4
esq = (x >= c - h) & (x < c)
dir_ = (x >= c) & (x <= c + h)

def slope_local(x_sub, y_sub):
    X = np.column_stack([np.ones(len(x_sub)), x_sub - c])
    coef = lstsq(X, y_sub, rcond=None)[0]
    return coef[1]  # slope

# Kink no tratamento
slope_b_esq = slope_local(x[esq], b[esq])
slope_b_dir = slope_local(x[dir_], b[dir_])
# Kink no resultado
slope_y_esq = slope_local(x[esq], Y[esq])
slope_y_dir = slope_local(x[dir_], Y[dir_])

rkd = (slope_y_dir - slope_y_esq) / (slope_b_dir - slope_b_esq + 1e-12)

print("=" * 70)
print("REGRESSION KINK DESIGN (RKD)")
print("Fonte: Cap. 34, pp. 326–333")
print("Autor: Luiz Tiago Wilcke")
print("=" * 70)
print(f"Delta slope Y: {(slope_y_dir - slope_y_esq):.3f}")
print(f"Delta slope b (tratamento): {(slope_b_dir - slope_b_esq):.3f}")
print(f"RKD estimado: {rkd:.3f}")
print(f"Tau verdadeiro: {tau_true:.1f}")
