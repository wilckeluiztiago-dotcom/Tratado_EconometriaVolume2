# Modelo 14 – MCMC Bayesiano (Gibbs para regressão linear)
# Autor: Luiz Tiago Wilcke
import numpy as np

np.random.seed(42)
n = 200
X = np.column_stack([np.ones(n), np.random.normal(0,1,n)])
beta_v = np.array([1.0, 2.0])
y = X @ beta_v + np.random.normal(0, 1, n)

# Gibbs sampler: prior N(0,100) para beta, IG(0.01,0.01) para sigma2
n_iter, burn = 3000, 1000
beta_chain = np.zeros((n_iter, 2))
sig2 = 1.0
beta = np.array([0.,0.])
XtX = X.T @ X
for i in range(n_iter):
    # beta | sig2, y
    V = np.linalg.inv(XtX/sig2 + np.eye(2)/100)
    m = V @ (X.T @ y / sig2)
    beta = np.random.multivariate_normal(m, V)
    # sig2 | beta, y
    resid = y - X @ beta
    a = 0.01 + n/2
    b = 0.01 + 0.5*np.sum(resid**2)
    sig2 = 1 / np.random.gamma(a, 1/b)
    beta_chain[i] = beta

post = beta_chain[burn:]
print("="*60)
print("MCMC BAYESIANO – Gibbs Sampler (Regressão Linear)")
print("Autor: Luiz Tiago Wilcke")
print("="*60)
print(f"Média a posteriori beta: {post.mean(axis=0).round(3)}")
print(f"IC 95% beta0: {np.percentile(post[:,0],[2.5,97.5]).round(3)}")
print(f"IC 95% beta1: {np.percentile(post[:,1],[2.5,97.5]).round(3)}")
print(f"Verdadeiro: {beta_v}")
