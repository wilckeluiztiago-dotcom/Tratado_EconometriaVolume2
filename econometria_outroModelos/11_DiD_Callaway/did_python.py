# ============================================================
# Modelo 11 – Diferenças em Diferenças: TWFE vs Callaway-Sant'Anna
# Fonte: Cap. 31, pp. 297–305 | Callaway-Sant'Anna: pp. 300–301
# Autor: Luiz Tiago Wilcke
# ============================================================

import numpy as np
import pandas as pd

np.random.seed(42)
N, T = 120, 10
# Adoção escalonada: coortes em t=4, 6, 8
coorte = np.array([4]*40 + [6]*40 + [8]*20 + [0]*20)  # 0 = nunca tratado
efeito_true = {4: 2.0, 6: 1.5, 8: 1.0}

linhas = []
for i in range(N):
    alpha_i = np.random.normal(0, 1)
    for t in range(1, T+1):
        tratado = 1 if (coorte[i] > 0 and t >= coorte[i]) else 0
        tau = efeito_true.get(coorte[i], 0) if tratado else 0
        y = alpha_i + 0.3*t + tau + np.random.normal(0, 1)
        linhas.append({"id": i, "tempo": t, "coorte": coorte[i],
                       "tratado": tratado, "y": y})
df = pd.DataFrame(linhas)

# TWFE
from numpy.linalg import lstsq
dummies_i = pd.get_dummies(df["id"], prefix="id", drop_first=True)
dummies_t = pd.get_dummies(df["tempo"], prefix="t", drop_first=True)
X_twfe = np.column_stack([df["tratado"].values,
                          dummies_i.values, dummies_t.values])
beta_twfe = lstsq(X_twfe, df["y"].values, rcond=None)[0][0]

# Callaway-Sant'Anna: ATT por coorte e tempo, controles = ainda não tratados / nunca
def att_cs(df, g, t):
    """ATT(g,t) usando nunca tratados + ainda não tratados como controle."""
    if t < g:
        return np.nan
    # Tratado: coorte g
    yg_t = df[(df["coorte"] == g) & (df["tempo"] == t)]["y"].mean()
    yg_pre = df[(df["coorte"] == g) & (df["tempo"] == g - 1)]["y"].mean()
    # Controle: nunca (coorte 0) ou coorte > t
    ctrl = df[(df["coorte"] == 0) | (df["coorte"] > t)]
    yc_t = ctrl[ctrl["tempo"] == t]["y"].mean()
    yc_pre = ctrl[ctrl["tempo"] == g - 1]["y"].mean()
    return (yg_t - yg_pre) - (yc_t - yc_pre)

atts = {}
for g in [4, 6, 8]:
    for t in range(g, T+1):
        atts[(g, t)] = att_cs(df, g, t)

# Agregação simples (média dos ATT)
att_medio = np.nanmean(list(atts.values()))

print("=" * 70)
print("DiD – TWFE vs CALLAWAY-SANT'ANNA (2021)")
print("Fonte: Cap. 31, pp. 297–305 | CS: pp. 300–301")
print("Autor: Luiz Tiago Wilcke")
print("=" * 70)
print(f"TWFE (pode ser viesado com adoção escalonada): {beta_twfe:.3f}")
print(f"ATT médio Callaway-Sant'Anna: {att_medio:.3f}")
print("ATT(g,t) por coorte:")
for (g, t), v in sorted(atts.items()):
    print(f"  coorte={g}, t={t}: {v:.3f}")
