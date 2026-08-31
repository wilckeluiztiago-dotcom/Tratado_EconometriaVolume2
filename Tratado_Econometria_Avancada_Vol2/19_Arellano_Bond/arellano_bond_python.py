# Modelo 19 – Arellano-Bond (GMM em diferenças) simplificado
# Autor: Luiz Tiago Wilcke
import numpy as np

np.random.seed(42)
N, T = 100, 8
# y_it = rho y_{i,t-1} + beta x_it + a_i + e_it
rho_v, beta_v = 0.6, 1.0
a = np.random.normal(0, 1, N)
x = np.random.normal(0, 1, (N, T))
y = np.zeros((N, T))
y[:,0] = a + np.random.normal(0,1,N)
for t in range(1,T):
    y[:,t] = rho_v * y[:,t-1] + beta_v * x[:,t] + a + np.random.normal(0,1,N)

# Diferenças e instrumentos (lags)
# Delta y_it = rho Delta y_{i,t-1} + beta Delta x_it + Delta e
# Instrumentos: y_{i,t-2}, y_{i,t-3}, ...
dy = y[:,1:] - y[:,:-1]
dx = x[:,1:] - x[:,:-1]
# Usar y_{t-2} como instrumento para Delta y_{t-1}
# Empilhar para t=2..T-1 (índices 1..)
Y = []; Xmat = []; Zlist = []
for t in range(2, T):  # t a partir de 3 em 1-index
    # Delta y_t depende de Delta y_{t-1}, instrumento y_{t-2}
    Y.append(dy[:,t-1])
    Xmat.append(np.column_stack([dy[:,t-2], dx[:,t-1]]))
    # instrumentos: y até t-2
    Zlist.append(y[:, :t-1])

# GMM 1 etapa simplificado (apenas um lag de instrumento)
Y_stack = np.concatenate(Y)
X_stack = np.vstack(Xmat)
Z_stack = np.vstack([np.column_stack([z, np.ones(N)]) for z in Zlist])  # simplificado

# 2SLS aproximado
# Para simplicidade: OLS em diferenças (viesado) e menção a GMM
from numpy.linalg import lstsq
theta_ols = lstsq(X_stack, Y_stack, rcond=None)[0]
print("="*60)
print("ARELLANO-BOND (GMM em diferenças) – aproximação")
print("Autor: Luiz Tiago Wilcke")
print("="*60)
print(f"rho OLS-dif (viesado): {theta_ols[0]:.3f} (verdadeiro={rho_v})")
print(f"beta OLS-dif: {theta_ols[1]:.3f} (verdadeiro={beta_v})")
print("Nota: em produção use linearmodels.PanelOLS ou pydynpd / R plm")
