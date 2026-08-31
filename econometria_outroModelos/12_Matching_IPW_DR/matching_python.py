# ============================================================
# Modelo 12 – Matching, IPW e Estimador Duplamente Robusto
# Fonte: Cap. 32, pp. 306–314 | DR: p. 308
# Autor: Luiz Tiago Wilcke
# ============================================================

import numpy as np
from scipy.spatial.distance import cdist

np.random.seed(42)
n = 1000
X = np.random.normal(0, 1, (n, 3))
ps_true = 1 / (1 + np.exp(-(0.5*X[:,0] - 0.3*X[:,1])))
D = (np.random.rand(n) < ps_true).astype(float)
tau_true = 2.0
Y = 1 + tau_true*D + 0.8*X[:,0] + 0.4*X[:,2] + np.random.normal(0, 1, n)

# Propensity score (logit MQ)
from scipy.optimize import minimize
def logit_negll(b, D, X):
    xb = X @ b
    p = 1/(1+np.exp(-xb))
    p = np.clip(p, 1e-6, 1-1e-6)
    return -np.sum(D*np.log(p)+(1-D)*np.log(1-p))
Xb = np.column_stack([np.ones(n), X])
res = minimize(logit_negll, np.zeros(4), args=(D, Xb), method="BFGS")
ps = 1/(1+np.exp(-(Xb @ res.x)))
ps = np.clip(ps, 0.05, 0.95)

# IPW (ATE)
ate_ipw = np.mean(D*Y/ps - (1-D)*Y/(1-ps))

# Matching 1:1 (nearest neighbor no PS)
trat = np.where(D==1)[0]
ctrl = np.where(D==0)[0]
dist = cdist(ps[trat].reshape(-1,1), ps[ctrl].reshape(-1,1))
match_idx = dist.argmin(axis=1)
ate_match = (Y[trat] - Y[ctrl[match_idx]]).mean()

# Duplamente robusto (AIPW)
# mu1, mu0 via regressão linear
mu1_coef = np.linalg.lstsq(Xb[D==1], Y[D==1], rcond=None)[0]
mu0_coef = np.linalg.lstsq(Xb[D==0], Y[D==0], rcond=None)[0]
mu1 = Xb @ mu1_coef
mu0 = Xb @ mu0_coef
ate_dr = np.mean(mu1 - mu0 + D*(Y-mu1)/ps - (1-D)*(Y-mu0)/(1-ps))

print("=" * 70)
print("MATCHING / IPW / DUPLAMENTE ROBUSTO (AIPW)")
print("Fonte: Cap. 32, pp. 306–314 | DR: p. 308")
print("Autor: Luiz Tiago Wilcke")
print("=" * 70)
print(f"ATE IPW:              {ate_ipw:.3f}")
print(f"ATE Matching (1:1):   {ate_match:.3f}")
print(f"ATE Duplamente Robusto: {ate_dr:.3f}")
print(f"ATE verdadeiro:       {tau_true:.1f}")
