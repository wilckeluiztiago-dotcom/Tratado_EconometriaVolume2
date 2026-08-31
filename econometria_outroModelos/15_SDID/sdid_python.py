# ============================================================
# Modelo 15 – Synthetic Difference-in-Differences (SDID)
# Fonte: Cap. 35, pp. 334–342
# Autor: Luiz Tiago Wilcke
# ============================================================

import numpy as np
from scipy.optimize import minimize

np.random.seed(42)
N, T, T0 = 20, 25, 15
# Unidade 0 tratada após T0
F = np.cumsum(np.random.normal(0, 0.3, T))
Y = np.zeros((N, T))
for i in range(N):
    Y[i] = 10 + i*0.3 + F + np.random.normal(0, 1, T)
efeito = 4.0
Y[0, T0:] += efeito

# Pesos de unidade (pré-tratamento): minimizar ||Y0_pre - w' Yj_pre||
Y_pre = Y[:, :T0]
y0_pre = Y_pre[0]
Yj_pre = Y_pre[1:]

def obj_w(w):
    return np.sum((y0_pre - w @ Yj_pre)**2)
cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
bounds = [(0, 1)] * (N-1)
w0 = np.ones(N-1)/(N-1)
res_w = minimize(obj_w, w0, method="SLSQP", bounds=bounds, constraints=cons)
omega = res_w.x

# Pesos de tempo (unidades de controle)
Y_ctrl = Y[1:]
y_pre_mean = Y_ctrl[:, :T0].mean(axis=0)
y_post_mean = Y_ctrl[:, T0:].mean(axis=0)

def obj_lambda(lam):
    # aproximação: alinhar média pré ponderada com pós
    return np.sum((lam @ y_pre_mean - y_post_mean.mean())**2)
# simplificação: lambda uniforme no pós e otimizado no pré
lambda_pre = np.ones(T0)/T0

# SDID: diferença ponderada
y0_post = Y[0, T0:].mean()
y0_pre_w = lambda_pre @ Y[0, :T0]
yc_post = omega @ Y[1:, T0:].mean(axis=1)
yc_pre = omega @ (Y[1:, :T0] @ lambda_pre)
sdid = (y0_post - y0_pre_w) - (yc_post - yc_pre)

print("=" * 70)
print("SYNTHETIC DIFFERENCE-IN-DIFFERENCES (SDID)")
print("Fonte: Cap. 35, pp. 334–342")
print("Autor: Luiz Tiago Wilcke")
print("=" * 70)
print(f"SDID estimado: {sdid:.3f}")
print(f"Efeito verdadeiro: {efeito:.1f}")
print(f"Pesos de unidade (top 5): {np.sort(omega)[::-1][:5].round(3)}")
