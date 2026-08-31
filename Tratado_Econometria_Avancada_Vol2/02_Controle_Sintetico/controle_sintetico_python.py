# ============================================================
# Modelo 02 – Controle Sintético (Synthetic Control)
# Tratado de Econometria Avançada – Volume II
# Autor: Luiz Tiago Wilcke
# ============================================================

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

np.random.seed(123)

# ------------------------------------------------------------
# 1. Dados sintéticos
# ------------------------------------------------------------
n_unidades = 11          # 1 tratada + 10 doadoras
n_periodos = 30
periodo_tratamento = 20  # T0 = 20

# Unidades: 0 = tratada, 1..10 = doadoras
unidades = [f"unidade_{i}" for i in range(n_unidades)]
periodos = np.arange(1, n_periodos + 1)

# Trajetórias base (fatores comuns + idiossincrático)
fator_comum = np.cumsum(np.random.normal(0, 0.3, n_periodos))
resultado = np.zeros((n_unidades, n_periodos))

for i in range(n_unidades):
    nivel = 50 + i * 2 + np.random.normal(0, 1)
    resultado[i, :] = nivel + fator_comum + np.random.normal(0, 1.5, n_periodos)

# Efeito do tratamento a partir de T0+1
efeito_verdadeiro = 8.0
resultado[0, periodo_tratamento:] += efeito_verdadeiro

# DataFrame em formato longo
linhas = []
for i, u in enumerate(unidades):
    for t, p in enumerate(periodos):
        linhas.append({
            "unidade": u,
            "periodo": p,
            "resultado": resultado[i, t],
            "tratada": 1 if i == 0 else 0,
            "pos_tratamento": 1 if (i == 0 and p > periodo_tratamento) else 0
        })
dados = pd.DataFrame(linhas)

# ------------------------------------------------------------
# 2. Função de otimização dos pesos do controle sintético
# ------------------------------------------------------------
def otimizar_pesos(Y_tratada_pre, Y_doadoras_pre):
    """
    Minimiza a distância pré-tratamento entre a unidade tratada
    e a combinação convexa das doadoras.
    """
    J = Y_doadoras_pre.shape[0]

    def objetivo(w):
        sintetic = w @ Y_doadoras_pre
        return np.sum((Y_tratada_pre - sintetic)**2)

    restricoes = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    limites = [(0, 1) for _ in range(J)]
    w0 = np.ones(J) / J

    res = minimize(objetivo, w0, method="SLSQP",
                   bounds=limites, constraints=restricoes)
    return res.x

# ------------------------------------------------------------
# 3. Estimação
# ------------------------------------------------------------
# Matrizes pré-tratamento
Y_tratada = dados[dados["unidade"] == "unidade_0"]["resultado"].values
Y_doadoras = np.vstack([
    dados[dados["unidade"] == u]["resultado"].values
    for u in unidades[1:]
])

Y_tratada_pre = Y_tratada[:periodo_tratamento]
Y_doadoras_pre = Y_doadoras[:, :periodo_tratamento]

pesos = otimizar_pesos(Y_tratada_pre, Y_doadoras_pre)
controle_sintetico = pesos @ Y_doadoras

# Efeito do tratamento
efeito = Y_tratada - controle_sintetico

print("=" * 60)
print("CONTROLE SINTÉTICO (SYNTHETIC CONTROL)")
print("Autor: Luiz Tiago Wilcke")
print("=" * 60)
print("Pesos ótimos das unidades doadoras:")
for i, w in enumerate(pesos):
    if w > 0.01:
        print(f"  {unidades[i+1]}: {w:.4f}")
print(f"\nEfeito médio pós-tratamento estimado: {efeito[periodo_tratamento:].mean():.3f}")
print(f"Efeito verdadeiro (simulado): {efeito_verdadeiro:.1f}")

# ------------------------------------------------------------
# 4. Gráfico
# ------------------------------------------------------------
plt.figure(figsize=(11, 5))
plt.plot(periodos, Y_tratada, "b-", linewidth=2, label="Unidade tratada")
plt.plot(periodos, controle_sintetico, "r--", linewidth=2, label="Controle sintético")
plt.axvline(periodo_tratamento, color="gray", linestyle=":", label="Início do tratamento")
plt.title("Controle Sintético – Resultado Observado vs. Sintético\nLuiz Tiago Wilcke")
plt.xlabel("Período")
plt.ylabel("Resultado")
plt.legend()
plt.tight_layout()
plt.savefig("02_Controle_Sintetico_resultado.png", dpi=120)
plt.close()
print("\nGráfico salvo: 02_Controle_Sintetico_resultado.png")
