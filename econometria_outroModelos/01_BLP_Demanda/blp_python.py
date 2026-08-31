# ============================================================
# Modelo 01 – BLP (Berry, Levinsohn & Pakes, 1995)
# Demanda estrutural com coeficientes aleatórios + inversão de shares
# Fonte: Tratado de Econometria Avançada – Vol. II, Cap. 14, pp. 127–138
#         Especialmente: Algoritmo de inversão de contração (pp. 129–130)
# Autor: Luiz Tiago Wilcke
# ============================================================

import numpy as np
from scipy.optimize import minimize
from numpy.linalg import inv, norm

np.random.seed(42)

# ------------------------------------------------------------
# 1. Dados sintéticos de mercado (J produtos, T mercados)
# ------------------------------------------------------------
J, T, n_cons = 5, 30, 200   # produtos, mercados, consumidores simulados
# Características: constante, preço, qualidade
X = np.random.uniform(0, 1, size=(T, J, 3))
X[:, :, 0] = 1.0
# Preços endógenos
custo = 0.5 + 0.3 * X[:, :, 2] + np.random.normal(0, 0.1, (T, J))
preco = custo + 0.4 + np.random.normal(0, 0.15, (T, J))
X[:, :, 1] = preco

# Preferências verdadeiras: beta ~ N(b, Sigma)
b_true = np.array([-1.5, -2.0, 1.0])   # preço tem coeficiente negativo
Sigma_true = np.diag([0.3, 0.5, 0.4])
xi_true = np.random.normal(0, 0.3, (T, J))  # choque de demanda

def simular_shares(X, b, Sigma, xi, n_draws=n_cons):
    """Simula shares de mercado via mixed logit."""
    T, J, K = X.shape
    shares = np.zeros((T, J))
    for t in range(T):
        draws = np.random.multivariate_normal(b, Sigma, n_draws)  # (n, K)
        util = draws @ X[t].T + xi[t]  # (n, J)
        util = np.hstack([util, np.zeros((n_draws, 1))])  # outside option
        expu = np.exp(util - util.max(axis=1, keepdims=True))
        p = expu / expu.sum(axis=1, keepdims=True)
        shares[t] = p[:, :J].mean(axis=0)
    return shares

shares = simular_shares(X, b_true, Sigma_true, xi_true)

# ------------------------------------------------------------
# 2. Inversão de BLP (contração de ponto fixo)
# ------------------------------------------------------------
def blp_contraction(shares, X, b, Sigma, tol=1e-8, maxit=500, n_draws=150):
    """
    Para cada mercado, encontra delta_j tal que shares preditos = observados.
    delta_j = x_j'b + xi_j  (parte não aleatória da utilidade média).
    """
    T, J, K = X.shape
    delta = np.log(shares + 1e-12) - np.log(1 - shares.sum(axis=1, keepdims=True) + 1e-12)
    draws = np.random.multivariate_normal(np.zeros(K), Sigma, n_draws)

    for t in range(T):
        d = delta[t].copy()
        for it in range(maxit):
            # utilidade com heterogeneidade
            mu = draws @ X[t].T          # (n_draws, J)
            util = d + mu
            util_full = np.hstack([util, np.zeros((n_draws, 1))])
            expu = np.exp(util_full - util_full.max(axis=1, keepdims=True))
            p = expu / expu.sum(axis=1, keepdims=True)
            s_hat = p[:, :J].mean(axis=0)
            d_new = d + np.log(shares[t] + 1e-12) - np.log(s_hat + 1e-12)
            if norm(d_new - d) < tol:
                d = d_new
                break
            d = d_new
        delta[t] = d
    return delta

# ------------------------------------------------------------
# 3. GMM do segundo estágio (linear em delta)
#    delta = X beta + xi ; E[Z' xi] = 0
# ------------------------------------------------------------
# Instrumentos: características próprias + somas das características dos rivais
Z_list = []
for t in range(T):
    rivais = []
    for j in range(J):
        outros = np.delete(X[t], j, axis=0).sum(axis=0)
        rivais.append(outros)
    Z_t = np.hstack([X[t], np.array(rivais)])  # (J, 2K)
    Z_list.append(Z_t)
Z = np.vstack(Z_list)          # (T*J, 2K)
delta = blp_contraction(shares, X, b_true, Sigma_true)
delta_vec = delta.ravel()
X_vec = X.reshape(T * J, -1)

# 2SLS / GMM linear
P_Z = Z @ inv(Z.T @ Z) @ Z.T
beta_hat = inv(X_vec.T @ P_Z @ X_vec) @ (X_vec.T @ P_Z @ delta_vec)
xi_hat = delta_vec - X_vec @ beta_hat

print("=" * 70)
print("BLP – DEMANDA ESTRUTURAL COM COEFICIENTES ALEATÓRIOS")
print("Fonte: Cap. 14, pp. 127–138 | Inversão de contração: pp. 129–130")
print("Autor: Luiz Tiago Wilcke")
print("=" * 70)
print(f"Beta estimado (const, preco, qualidade): {beta_hat.round(3)}")
print(f"Beta verdadeiro aproximado:             {b_true}")
print(f"Correlação xi_hat vs xi_true: {np.corrcoef(xi_hat, xi_true.ravel())[0,1]:.3f}")
print(f"Shares médios observados: {shares.mean(axis=0).round(3)}")
