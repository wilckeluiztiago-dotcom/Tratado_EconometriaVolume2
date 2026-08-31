# ============================================================
# Modelo 18 – Estimador GPV de Leilões (Guerre, Perrigne, Vuong 2000)
# Fonte: Cap. 18, pp. 173–180 | GPV: pp. 174–175
# Autor: Luiz Tiago Wilcke
# ============================================================

import numpy as np

np.random.seed(42)
# Leilão de primeiro preço, IPV, n=3 licitantes
n_bidders = 3
n_auctions = 800
# Valores ~ U[0,1]; lance de equilíbrio: b(v) = (n-1)/n * v  (uniforme)
valores = np.random.uniform(0, 1, (n_auctions, n_bidders))
lances = (n_bidders - 1) / n_bidders * valores
# Observamos apenas lances
lances_obs = lances.ravel()

# GPV: recuperar valores a partir de lances
# v = b + G(b)/((n-1) g(b))
# Estimar G e g por kernel nos lances
from scipy.stats import gaussian_kde
kde = gaussian_kde(lances_obs, bw_method="scott")
# Grade
b_grid = np.linspace(lances_obs.min()+0.01, lances_obs.max()-0.01, 100)
g_hat = kde(b_grid)
# CDF empírica suavizada
G_hat = np.array([np.mean(lances_obs <= b) for b in b_grid])
# Valores recuperados
v_hat = b_grid + G_hat / ((n_bidders - 1) * g_hat + 1e-12)
v_hat = np.clip(v_hat, 0, 1.5)

# Comparar distribuição
print("=" * 70)
print("GPV – LEILÕES ESTRUTURAIS (Guerre-Perrigne-Vuong 2000)")
print("Fonte: Cap. 18, pp. 173–180 | Estimador não-paramétrico: pp. 174–175")
print("Autor: Luiz Tiago Wilcke")
print("=" * 70)
print(f"Lance médio: {lances_obs.mean():.3f}")
print(f"Valor recuperado médio (GPV): {v_hat.mean():.3f}")
print(f"Valor verdadeiro médio: {valores.mean():.3f}")
print(f"Correlação lance vs valor verdadeiro: {np.corrcoef(lances.ravel(), valores.ravel())[0,1]:.3f}")
