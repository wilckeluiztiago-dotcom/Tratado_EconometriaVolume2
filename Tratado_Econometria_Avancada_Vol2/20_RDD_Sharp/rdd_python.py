# Modelo 20 – Sharp RDD
# Autor: Luiz Tiago Wilcke
import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(42)
n = 1000
# Running variable
x = np.random.uniform(-1, 1, n)
# Tratamento: x >= 0
D = (x >= 0).astype(float)
# Resultado com salto de 2 no limiar
y = 1 + 0.5*x + 2*D + np.random.normal(0, 1, n)

# Estimação local linear (bandwidth fixo)
h = 0.3
esq = (x >= -h) & (x < 0)
dir = (x >= 0) & (x <= h)

# Lado esquerdo
Xe = np.column_stack([np.ones(esq.sum()), x[esq]])
be = np.linalg.lstsq(Xe, y[esq], rcond=None)[0]
# Lado direito
Xd = np.column_stack([np.ones(dir.sum()), x[dir]])
bd = np.linalg.lstsq(Xd, y[dir], rcond=None)[0]

efeito = (bd[0] + bd[1]*0) - (be[0] + be[1]*0)
print("="*60)
print("SHARP RDD – Regressão de Descontinuidade Nítida")
print("Autor: Luiz Tiago Wilcke")
print("="*60)
print(f"Efeito local estimado no limiar: {efeito:.3f}")
print(f"Efeito verdadeiro: 2.0")
print(f"Bandwidth: {h}")
print(f"Obs. à esquerda: {esq.sum()}, à direita: {dir.sum()}")
