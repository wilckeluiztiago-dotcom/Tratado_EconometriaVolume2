# ============================================================
# Modelo 03 – Double Machine Learning (DML)
# Tratado de Econometria Avançada – Volume II
# Autor: Luiz Tiago Wilcke
# ============================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ------------------------------------------------------------
# 1. Dados sintéticos
# ------------------------------------------------------------
n = 2000
# Covariáveis de alta dimensão
X = np.random.normal(0, 1, size=(n, 10))
# Tratamento endógeno
tratamento = (0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.normal(0, 1, n) > 0).astype(float)
# Resultado com efeito causal verdadeiro = 2.0
efeito_verdadeiro = 2.0
resultado = efeito_verdadeiro * tratamento + 1.5 * X[:, 0] + 0.8 * X[:, 2]**2 + np.random.normal(0, 1, n)

dados = pd.DataFrame(X, columns=[f"covariavel_{i+1}" for i in range(10)])
dados["tratamento"] = tratamento
dados["resultado"] = resultado

# ------------------------------------------------------------
# 2. Double Machine Learning com Cross-Fitting
# ------------------------------------------------------------
def double_ml_parcialmente_linear(Y, D, X, n_folds=5):
    """
    DML para o modelo parcialmente linear:
        Y = theta * D + g(X) + e
        D = m(X) + v
    Ortogonalização de Neyman + cross-fitting.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    residuos_Y = np.zeros(len(Y))
    residuos_D = np.zeros(len(D))

    for train_idx, test_idx in kf.split(X):
        # Modelos de nuisance
        modelo_g = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        modelo_m = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)

        modelo_g.fit(X[train_idx], Y[train_idx])
        modelo_m.fit(X[train_idx], D[train_idx])

        residuos_Y[test_idx] = Y[test_idx] - modelo_g.predict(X[test_idx])
        residuos_D[test_idx] = D[test_idx] - modelo_m.predict(X[test_idx])

    # Regressão residual (estimador de Neyman ortogonal)
    theta = np.sum(residuos_D * residuos_Y) / np.sum(residuos_D**2)
    # Erro-padrão robusto
    n = len(Y)
    psi = residuos_D * (residuos_Y - theta * residuos_D)
    se = np.sqrt(np.mean(psi**2) / n) / (np.mean(residuos_D**2))
    return theta, se

# ------------------------------------------------------------
# 3. Estimação
# ------------------------------------------------------------
Y = dados["resultado"].values
D = dados["tratamento"].values
X = dados[[f"covariavel_{i+1}" for i in range(10)]].values

theta_hat, se_hat = double_ml_parcialmente_linear(Y, D, X)

print("=" * 60)
print("DOUBLE MACHINE LEARNING (DML) – MODELO PARCIALMENTE LINEAR")
print("Autor: Luiz Tiago Wilcke")
print("=" * 60)
print(f"Efeito causal estimado (theta): {theta_hat:.4f}")
print(f"Erro-padrão: {se_hat:.4f}")
print(f"IC 95%: [{theta_hat - 1.96*se_hat:.4f}, {theta_hat + 1.96*se_hat:.4f}]")
print(f"Efeito verdadeiro (simulado): {efeito_verdadeiro:.1f}")
print(f"Número de observações: {n}")
print(f"Número de covariáveis: 10")
