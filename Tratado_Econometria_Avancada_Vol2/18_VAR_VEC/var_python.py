# Modelo 18 – VAR e Cointegração (Johansen simplificado)
# Autor: Luiz Tiago Wilcke
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen

np.random.seed(42)
T = 200
# Dois processos cointegrados: x_t = x_{t-1} + e, y_t = 0.5 x_t + u
e = np.random.normal(0,1,T)
u = np.random.normal(0,1,T)
x = np.cumsum(e)
y = 0.5 * x + u
dados = np.column_stack([y, x])

# VAR em níveis
modelo = VAR(dados)
res = modelo.fit(2)
print("="*60)
print("VAR(2) e TESTE DE COINTEGRAÇÃO DE JOHANSEN")
print("Autor: Luiz Tiago Wilcke")
print("="*60)
print(res.summary())

# Johansen
joh = coint_johansen(dados, det_order=0, k_ar_diff=1)
print("\nEstatística do traço (Johansen):", joh.lr1)
print("Valores críticos (90%,95%,99%):", joh.cvt)
