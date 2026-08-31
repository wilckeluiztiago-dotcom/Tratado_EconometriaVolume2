# Modelo 08 – DSGE Novo-Keynesiano (3 equações, simulação)
# Autor: Luiz Tiago Wilcke
import numpy as np
import pandas as pd

np.random.seed(42)
T = 100
# Parâmetros canônicos
sigma, kappa, phi_pi, phi_y, rho = 1.0, 0.1, 1.5, 0.5, 0.8
# Simulação sob expectativas racionais (forma reduzida simplificada)
y_gap = np.zeros(T)
pi = np.zeros(T)
i = np.zeros(T)
shock = np.random.normal(0, 0.5, T)
for t in range(1, T):
    # Aproximação: y_t = rho*y_{t-1} + shock (IS + política)
    y_gap[t] = 0.7 * y_gap[t-1] - 0.3*(i[t-1]-pi[t-1]) + shock[t]
    pi[t] = 0.6 * pi[t-1] + kappa * y_gap[t] + 0.1*np.random.normal()
    i[t] = phi_pi * pi[t] + phi_y * y_gap[t] + 0.2*np.random.normal()

dados = pd.DataFrame({"periodo": range(1,T+1), "hiato_produto": y_gap,
                      "inflacao": pi, "taxa_juros": i})
print("="*60)
print("DSGE NOVO-KEYNESIANO (simulação 3 equações)")
print("Autor: Luiz Tiago Wilcke")
print("="*60)
print(dados.describe().round(3))
print("\nParâmetros: sigma=1, kappa=0.1, phi_pi=1.5, phi_y=0.5")
