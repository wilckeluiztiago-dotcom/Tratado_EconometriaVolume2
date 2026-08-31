# ============================================================
# Modelo 02 – NFXP (Nested Fixed Point) de Rust (1987)
# Escolha discreta dinâmica: substituição de ônibus
# Fonte: Cap. 15, pp. 139–150 | NFXP: pp. 140–141
# Autor: Luiz Tiago Wilcke
# ============================================================

import numpy as np
from scipy.optimize import minimize

np.random.seed(42)

# Estados: milhagem discretizada (0..n_states-1)
n_states = 20
n_actions = 2  # 0=manter, 1=substituir
beta = 0.95    # fator de desconto
# Custo de manutenção linear no estado; custo fixo de substituição
theta_true = np.array([0.5, 4.0])  # (custo_manutencao, RC)

def utilidade(estado, acao, theta):
    if acao == 1:  # substituir
        return -theta[1]
    return -theta[0] * estado  # manter

def transicao(estado, acao, n_states):
    """Probabilidades de transição do estado."""
    p = np.zeros(n_states)
    if acao == 1:
        p[0] = 1.0  # volta a zero
    else:
        # milhagem aumenta 0, 1 ou 2 com probs fixas
        for d, pr in [(0, 0.3), (1, 0.5), (2, 0.2)]:
            s2 = min(estado + d, n_states - 1)
            p[s2] += pr
    return p

def fixed_point_EV(theta, beta, n_states, tol=1e-8, maxit=500):
    """Nested fixed point: resolve EV(s) = T_theta EV(s)."""
    EV = np.zeros(n_states)
    for it in range(maxit):
        EV_new = np.zeros(n_states)
        for s in range(n_states):
            v = np.zeros(n_actions)
            for a in range(n_actions):
                p = transicao(s, a, n_states)
                v[a] = utilidade(s, a, theta) + beta * p @ EV
            # log-sum-exp (Gumbel)
            m = v.max()
            EV_new[s] = m + np.log(np.exp(v - m).sum())
        if np.max(np.abs(EV_new - EV)) < tol:
            return EV_new
        EV = EV_new
    return EV

def choice_prob(estado, theta, EV, beta, n_states):
    v = np.zeros(n_actions)
    for a in range(n_actions):
        p = transicao(estado, a, n_states)
        v[a] = utilidade(estado, a, theta) + beta * p @ EV
    m = v.max()
    p = np.exp(v - m)
    return p / p.sum()

# Simular dados
EV_true = fixed_point_EV(theta_true, beta, n_states)
n_obs = 2000
estados = np.random.randint(0, n_states, n_obs)
acoes = np.array([np.random.choice(n_actions, p=choice_prob(s, theta_true, EV_true, beta, n_states))
                  for s in estados])

def neg_loglik(theta):
    if theta[0] <= 0 or theta[1] <= 0:
        return 1e10
    EV = fixed_point_EV(theta, beta, n_states)
    ll = 0.0
    for s, a in zip(estados, acoes):
        p = choice_prob(s, theta, EV, beta, n_states)
        ll += np.log(p[a] + 1e-12)
    return -ll

res = minimize(neg_loglik, [0.3, 3.0], method="Nelder-Mead",
               options={"maxiter": 80})
print("=" * 70)
print("NFXP – NESTED FIXED POINT (Rust 1987)")
print("Fonte: Cap. 15, pp. 139–150 | Algoritmo NFXP: pp. 140–141")
print("Autor: Luiz Tiago Wilcke")
print("=" * 70)
print(f"Theta estimado (c_manut, RC): {res.x.round(3)}")
print(f"Theta verdadeiro:             {theta_true}")
print(f"Log-lik: {-res.fun:.2f}")
