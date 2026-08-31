# Modelo 17 – Logit Misto (Mixed Logit) simplificado
# Autor: Luiz Tiago Wilcke
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize

np.random.seed(42)
n_indiv, n_alt, n_rep = 200, 3, 50
# Utilidade: V = beta_i * x , beta_i ~ N(b, sigma)
x = np.random.normal(0, 1, (n_indiv, n_alt))
b_v, sigma_v = 1.0, 0.8
beta_i = np.random.normal(b_v, sigma_v, n_indiv)
# Escolha
util = beta_i[:, None] * x
prob = np.exp(util - logsumexp(util, axis=1, keepdims=True))
escolha = np.array([np.random.choice(n_alt, p=prob[i]) for i in range(n_indiv)])

def sim_ll(theta):
    b, log_s = theta
    s = np.exp(log_s)
    draws = np.random.normal(b, s, (n_indiv, n_rep))
    ll = 0.0
    for i in range(n_indiv):
        u = draws[i,:,None] * x[i]
        p = np.exp(u - logsumexp(u, axis=1, keepdims=True))
        ll += np.log(p[:, escolha[i]].mean() + 1e-12)
    return -ll

res = minimize(sim_ll, [0.5, 0.0], method="Nelder-Mead")
print("="*60)
print("LOGIT MISTO (Mixed Logit – simulação de máxima verossimilhança)")
print("Autor: Luiz Tiago Wilcke")
print("="*60)
print(f"b estimado: {res.x[0]:.3f} (verdadeiro={b_v})")
print(f"sigma estimado: {np.exp(res.x[1]):.3f} (verdadeiro={sigma_v})")
