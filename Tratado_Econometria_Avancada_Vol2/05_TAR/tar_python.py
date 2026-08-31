# ============================================================
# Modelo 05 – TAR (Threshold Autoregressive)
# Autor: Luiz Tiago Wilcke
# ============================================================

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

np.random.seed(42)
n = 300
# Processo TAR: regime 1 se y_{t-1} <= limiar, regime 2 caso contrário
limiar_verdadeiro = 0.5
y = np.zeros(n)
y[0] = 0.0
for t in range(1, n):
    if y[t-1] <= limiar_verdadeiro:
        y[t] = 0.3 + 0.6 * y[t-1] + np.random.normal(0, 0.8)
    else:
        y[t] = -0.2 + 0.3 * y[t-1] + np.random.normal(0, 0.8)

dados = pd.DataFrame({"periodo": np.arange(1, n+1), "serie": y})

def ss_tar(limiar, serie):
    y = serie[1:]
    ylag = serie[:-1]
    regime1 = ylag <= limiar
    regime2 = ~regime1
    if regime1.sum() < 10 or regime2.sum() < 10:
        return 1e10
    # OLS por regime
    X1 = np.column_stack([np.ones(regime1.sum()), ylag[regime1]])
    X2 = np.column_stack([np.ones(regime2.sum()), ylag[regime2]])
    b1 = np.linalg.lstsq(X1, y[regime1], rcond=None)[0]
    b2 = np.linalg.lstsq(X2, y[regime2], rcond=None)[0]
    resid = np.zeros_like(y)
    resid[regime1] = y[regime1] - X1 @ b1
    resid[regime2] = y[regime2] - X2 @ b2
    return np.sum(resid**2)

# Busca do limiar
res = minimize_scalar(lambda c: ss_tar(c, y), bounds=(-2, 2), method="bounded")
limiar_est = res.x

print("=" * 60)
print("MODELO TAR (Threshold Autoregressive)")
print("Autor: Luiz Tiago Wilcke")
print("=" * 60)
print(f"Limiar estimado: {limiar_est:.4f}")
print(f"Limiar verdadeiro: {limiar_verdadeiro:.1f}")
print(f"Soma de quadrados residual: {res.fun:.2f}")
