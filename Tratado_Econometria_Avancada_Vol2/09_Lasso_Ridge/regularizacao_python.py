# Modelo 09 – Lasso / Ridge / Elastic Net
# Autor: Luiz Tiago Wilcke
import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n, p = 200, 50
X = np.random.normal(0, 1, (n, p))
beta_v = np.zeros(p); beta_v[:5] = [1.5, -1.2, 0.8, 0.5, -0.7]
y = X @ beta_v + np.random.normal(0, 1, n)
Xs = StandardScaler().fit_transform(X)

lasso = LassoCV(cv=5).fit(Xs, y)
ridge = RidgeCV(cv=5).fit(Xs, y)
enet  = ElasticNetCV(cv=5, l1_ratio=[0.1,0.5,0.9]).fit(Xs, y)

print("="*60)
print("REGULARIZAÇÃO – LASSO / RIDGE / ELASTIC NET")
print("Autor: Luiz Tiago Wilcke")
print("="*60)
print("Coeficientes verdadeiros (primeiros 5):", beta_v[:5])
print("Lasso  (não-zeros):", np.round(lasso.coef_[:5], 3), " | alpha=", round(lasso.alpha_,4))
print("Ridge:", np.round(ridge.coef_[:5], 3))
print("ElasticNet:", np.round(enet.coef_[:5], 3))
