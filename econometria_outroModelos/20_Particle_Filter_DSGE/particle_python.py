# ============================================================
# Modelo 20 – Filtro de Partículas (SIR) para DSGE Não-Linear
# Fonte: Cap. 41, pp. 391–397 | Filtragem de Partículas: pp. 392–393
# Autor: Luiz Tiago Wilcke
# ============================================================

import numpy as np

np.random.seed(42)

# Estado latente não-linear: x_t = 0.8 x_{t-1} + 0.1 x_{t-1}^3 + sig_x * e_t
# Observação: y_t = x_t + 0.05 x_t^2 + sig_y * u_t
T = 100
sig_x, sig_y = 0.3, 0.2
x_true = np.zeros(T)
y = np.zeros(T)
x_true[0] = 0.0
for t in range(1, T):
    x_true[t] = 0.8*x_true[t-1] + 0.1*x_true[t-1]**3 + sig_x*np.random.normal()
    y[t] = x_true[t] + 0.05*x_true[t]**2 + sig_y*np.random.normal()

# SIR Particle Filter
N_part = 500
particles = np.random.normal(0, 0.5, N_part)
weights = np.ones(N_part) / N_part
x_filt = np.zeros(T)

for t in range(T):
    # Propagação
    particles = 0.8*particles + 0.1*particles**3 + sig_x*np.random.normal(0, 1, N_part)
    # Pesos (verossimilhança)
    y_pred = particles + 0.05*particles**2
    log_w = -0.5 * ((y[t] - y_pred) / sig_y)**2
    log_w -= log_w.max()
    weights = np.exp(log_w)
    weights /= weights.sum()
    # Estimativa filtrada
    x_filt[t] = np.sum(weights * particles)
    # Reamostragem (sistemática)
    cum = np.cumsum(weights)
    u0 = np.random.uniform(0, 1/N_part)
    idx = np.searchsorted(cum, u0 + np.arange(N_part)/N_part)
    particles = particles[idx]
    weights = np.ones(N_part) / N_part

rmse = np.sqrt(np.mean((x_filt - x_true)**2))
print("=" * 70)
print("FILTRO DE PARTÍCULAS (SIR) – DSGE NÃO-LINEAR")
print("Fonte: Cap. 41, pp. 391–397 | Particle Filter: pp. 392–393")
print("Autor: Luiz Tiago Wilcke")
print("=" * 70)
print(f"RMSE filtro vs estado verdadeiro: {rmse:.4f}")
print(f"Desvio-padrão do estado: {x_true.std():.4f}")
print(f"Correlação filtrado vs verdadeiro: {np.corrcoef(x_filt, x_true)[0,1]:.3f}")
print(f"Número de partículas: {N_part}")
