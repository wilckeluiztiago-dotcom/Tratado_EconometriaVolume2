# Modelo 10 – Causal Forest (simplificado)
# Autor: Luiz Tiago Wilcke
import numpy as np
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)
n = 1000
X = np.random.normal(0, 1, (n, 5))
tratamento = (X[:,0] + np.random.normal(0,1,n) > 0).astype(float)
# Heterogeneidade: efeito = 1 + 0.5*X[:,1]
efeito_v = 1 + 0.5*X[:,1]
y = efeito_v * tratamento + X[:,2] + np.random.normal(0,1,n)

# Honest splitting simplificado: T-learner
rf0 = RandomForestRegressor(n_estimators=100, min_samples_leaf=20, random_state=42)
rf1 = RandomForestRegressor(n_estimators=100, min_samples_leaf=20, random_state=42)
rf0.fit(X[tratamento==0], y[tratamento==0])
rf1.fit(X[tratamento==1], y[tratamento==1])
cate = rf1.predict(X) - rf0.predict(X)

print("="*60)
print("CAUSAL FOREST (T-learner aproximado)")
print("Autor: Luiz Tiago Wilcke")
print("="*60)
print(f"CATE médio estimado: {cate.mean():.3f}")
print(f"CATE médio verdadeiro: {efeito_v.mean():.3f}")
print(f"Correlação CATE est vs verdadeiro: {np.corrcoef(cate, efeito_v)[0,1]:.3f}")
