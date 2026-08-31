# ============================================================
# Modelo 01 – Filtro de Kalman (Nível Local)
# Tratado de Econometria Avançada – Volume II
# Autor: Luiz Tiago Wilcke
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Geração de dados sintéticos
# ------------------------------------------------------------
np.random.seed(42)
n_obs = 120

# Estado latente: tendência (passeio aleatório)
tendencia_verdadeira = np.cumsum(np.random.normal(0, 0.6, n_obs)) + 10.0

# Observação: inflação observada = tendência + ruído de medição
ruido_medicao = np.random.normal(0, 1.2, n_obs)
inflacao_observada = tendencia_verdadeira + ruido_medicao

dados = pd.DataFrame({
    "periodo": np.arange(1, n_obs + 1),
    "inflacao_observada": inflacao_observada,
    "tendencia_verdadeira": tendencia_verdadeira
})

# ------------------------------------------------------------
# 2. Classe do Filtro de Kalman linear
# ------------------------------------------------------------
class FiltroKalmanNivelLocal:
    """
    Filtro de Kalman para o modelo de nível local:
        y_t = alpha_t + epsilon_t,  epsilon ~ N(0, H)
        alpha_t = alpha_{t-1} + eta_t, eta ~ N(0, Q)
    """
    def __init__(self, H=1.5, Q=0.5, a0=10.0, P0=4.0):
        self.H = H          # variância do erro de medição
        self.Q = Q          # variância do erro de transição
        self.a = a0         # estimativa inicial do estado
        self.P = P0         # variância inicial do estado

    def filtrar(self, serie_y):
        n = len(serie_y)
        estados_filtrados = np.zeros(n)
        variancias = np.zeros(n)
        inovacoes = np.zeros(n)

        for t in range(n):
            # --- Predição (a priori) ---
            a_prior = self.a
            P_prior = self.P + self.Q

            y_t = serie_y[t]
            if np.isnan(y_t):
                self.a = a_prior
                self.P = P_prior
            else:
                # --- Inovação ---
                v = y_t - a_prior
                F = P_prior + self.H

                # --- Ganho de Kalman ---
                K = P_prior / F

                # --- Atualização (a posteriori) ---
                self.a = a_prior + K * v
                self.P = (1 - K) * P_prior

                inovacoes[t] = v

            estados_filtrados[t] = self.a
            variancias[t] = self.P

        return estados_filtrados, variancias, inovacoes


# ------------------------------------------------------------
# 3. Estimação
# ------------------------------------------------------------
kf = FiltroKalmanNivelLocal(H=1.5, Q=0.5, a0=10.0, P0=4.0)
estados_est, var_est, inov = kf.filtrar(dados["inflacao_observada"].values)

dados["tendencia_filtrada"] = estados_est
dados["variancia_estado"] = var_est

# ------------------------------------------------------------
# 4. Resultados
# ------------------------------------------------------------
eqm = np.mean((dados["tendencia_verdadeira"] - dados["tendencia_filtrada"])**2)
print("=" * 60)
print("FILTRO DE KALMAN – MODELO DE NÍVEL LOCAL")
print("Autor: Luiz Tiago Wilcke")
print("=" * 60)
print(f"Número de observações: {n_obs}")
print(f"Erro Quadrático Médio (EQM): {eqm:.4f}")
print(f"Última estimativa do estado (tendência): {estados_est[-1]:.4f}")
print(f"Última variância do estado: {var_est[-1]:.4f}")
print("\nPrimeiras 8 linhas do resultado:")
print(dados[["periodo", "inflacao_observada", "tendencia_verdadeira",
             "tendencia_filtrada"]].head(8).to_string(index=False))

# ------------------------------------------------------------
# 5. Gráfico
# ------------------------------------------------------------
plt.figure(figsize=(11, 5))
plt.plot(dados["periodo"], dados["inflacao_observada"], "o", color="gray",
         alpha=0.5, markersize=4, label="Inflação observada")
plt.plot(dados["periodo"], dados["tendencia_verdadeira"], "--", color="black",
         linewidth=1.5, label="Tendência verdadeira (latente)")
plt.plot(dados["periodo"], dados["tendencia_filtrada"], color="blue",
         linewidth=1.8, label="Tendência filtrada (Kalman)")
plt.fill_between(dados["periodo"],
                 dados["tendencia_filtrada"] - 1.96 * np.sqrt(dados["variancia_estado"]),
                 dados["tendencia_filtrada"] + 1.96 * np.sqrt(dados["variancia_estado"]),
                 color="blue", alpha=0.15, label="IC 95%")
plt.title("Filtro de Kalman – Modelo de Nível Local\nLuiz Tiago Wilcke")
plt.xlabel("Período")
plt.ylabel("Inflação / Tendência")
plt.legend()
plt.tight_layout()
plt.savefig("01_Filtro_Kalman_resultado.png", dpi=120)
plt.close()
print("\nGráfico salvo: 01_Filtro_Kalman_resultado.png")
