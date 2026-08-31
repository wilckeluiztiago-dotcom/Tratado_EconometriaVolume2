# ============================================================
# Modelo 03 – CCP de Hotz e Miller (1993)
# Estimação de escolha discreta dinâmica via Conditional Choice Probabilities
# Fonte: Cap. 15, pp. 141–142
# Autor: Luiz Tiago Wilcke
# ============================================================

import numpy as np
from scipy.optimize import minimize

np.random.seed(42)
n_states, n_actions = 12, 2
beta, theta_true = 0.9, np.array([0.4, 3.5])

def util(s, a, th):
    return -th[1] if a == 1 else -th[0] * s

def P_trans(s, a, ns):
    p = np.zeros(ns)
    if a == 1:
        p[0] = 1.0
    else:
        for d, pr in [(0, 0.25), (1, 0.5), (2, 0.25)]:
            p[min(s + d, ns - 1)] += pr
    return p

# Gerar CCPs verdadeiras via NFXP interno
def get_ccp(th):
    EV = np.zeros(n_states)
    for _ in range(300):
        EVn = np.zeros(n_states)
        for s in range(n_states):
            v = [util(s, a, th) + beta * P_trans(s, a, n_states) @ EV for a in range(n_actions)]
            m = max(v)
            EVn[s] = m + np.log(sum(np.exp(np.array(v) - m)))
        if np.max(np.abs(EVn - EV)) < 1e-9:
            break
        EV = EVn
    ccp = np.zeros((n_states, n_actions))
    for s in range(n_states):
        v = [util(s, a, th) + beta * P_trans(s, a, n_states) @ EV for a in range(n_actions)]
        m = max(v)
        p = np.exp(np.array(v) - m)
        ccp[s] = p / p.sum()
    return ccp

ccp_true = get_ccp(theta_true)
# Dados: estados e ações
n_obs = 3000
estados = np.random.randint(0, n_states, n_obs)
acoes = np.array([np.random.choice(n_actions, p=ccp_true[s]) for s in estados])

# CCP não-paramétrica (frequências)
ccp_hat = np.zeros((n_states, n_actions))
for s in range(n_states):
    mask = estados == s
    if mask.sum() > 0:
        for a in range(n_actions):
            ccp_hat[s, a] = (acoes[mask] == a).mean()
    else:
        ccp_hat[s] = 1.0 / n_actions
ccp_hat = np.clip(ccp_hat, 0.01, 0.99)
ccp_hat /= ccp_hat.sum(axis=1, keepdims=True)

# Hotz-Miller: inversão CCP -> diferenças de valor
# V_a - V_0 = log(ccp_a) - log(ccp_0)
def hotz_miller_obj(th):
    # Construir valores relativos a partir das CCPs e utilidades paramétricas
    ll = 0.0
    for s, a in zip(estados, acoes):
        v = np.array([util(s, aa, th) for aa in range(n_actions)])
        # ajuste Hotz-Miller: adicionar log-CCP como proxy do valor futuro
        v = v + beta * np.log(ccp_hat[s] + 1e-12)  # simplificação pedagógica
        m = v.max()
        p = np.exp(v - m); p /= p.sum()
        ll += np.log(p[a] + 1e-12)
    return -ll

res = minimize(hotz_miller_obj, [0.2, 2.5], method="Nelder-Mead")
print("=" * 70)
print("CCP – HOTZ & MILLER (1993)")
print("Fonte: Cap. 15, pp. 141–142")
print("Autor: Luiz Tiago Wilcke")
print("=" * 70)
print(f"Theta estimado: {res.x.round(3)}")
print(f"Theta verdadeiro: {theta_true}")
print(f"CCP média (manter, substituir): {ccp_hat.mean(axis=0).round(3)}")
