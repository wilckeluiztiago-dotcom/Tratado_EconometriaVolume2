# ============================================================
# Modelo 13 – Controle Sintético Generalizado e Matrix Completion
# Fonte: Cap. 33, pp. 315–324 | Matrix Completion: pp. 317–318
# Autor: Luiz Tiago Wilcke
# ============================================================

import numpy as np
from numpy.linalg import svd

np.random.seed(42)
N, T = 25, 30
T0 = 20  # pré-tratamento
# Fatores latentes
F = np.random.normal(0, 1, (T, 3))
Lambda = np.random.normal(0, 1, (N, 3))
Y0 = Lambda @ F.T + np.random.normal(0, 0.5, (N, T))
# Tratamento na unidade 0 a partir de T0
efeito = 3.0
Y = Y0.copy()
Y[0, T0:] += efeito

# Matrix Completion via Soft-Impute (norma nuclear)
def soft_impute(M, mask, tau, maxit=100, tol=1e-5):
    """
    Completa matriz com valores faltantes (mask=0) penalizando norma nuclear.
    mask=1 onde observado.
    """
    Z = M.copy()
    Z[mask == 0] = 0
    for it in range(maxit):
        U, s, Vt = svd(Z, full_matrices=False)
        s_soft = np.maximum(s - tau, 0)
        Z_new = U @ np.diag(s_soft) @ Vt
        Z_new[mask == 1] = M[mask == 1]  # projeta nos observados
        if np.max(np.abs(Z_new - Z)) < tol:
            return Z_new
        Z = Z_new
    return Z

# Máscara: unidade 0 pós-tratamento é "faltante" para o contrafactual
mask = np.ones((N, T))
mask[0, T0:] = 0
Y_obs = Y.copy()
Y_obs[0, T0:] = 0  # remove valores tratados

Y_hat = soft_impute(Y_obs, mask, tau=5.0)
efeito_est = (Y[0, T0:] - Y_hat[0, T0:]).mean()

print("=" * 70)
print("MATRIX COMPLETION / CONTROLE SINTÉTICO GENERALIZADO")
print("Fonte: Cap. 33, pp. 315–324 | Norma nuclear: pp. 317–318")
print("Autor: Luiz Tiago Wilcke")
print("=" * 70)
print(f"Efeito médio pós-tratamento estimado: {efeito_est:.3f}")
print(f"Efeito verdadeiro: {efeito:.1f}")
print(f"RMSE predição pré-tratamento (unidade 0): {np.sqrt(np.mean((Y[0,:T0]-Y_hat[0,:T0])**2)):.3f}")
