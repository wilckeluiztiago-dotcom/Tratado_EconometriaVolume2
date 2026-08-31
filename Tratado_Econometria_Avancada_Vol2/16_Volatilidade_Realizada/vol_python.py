# Modelo 16 – Volatilidade Realizada e TSRV
# Autor: Luiz Tiago Wilcke
import numpy as np

np.random.seed(42)
# Simulação de preços de alta frequência com ruído de microestrutura
n_dias = 20
n_ticks = 390  # ~1 min
rv_classica = []
tsrv = []
for d in range(n_dias):
    # Processo latente (Browniano)
    sigma = 0.02
    dW = np.random.normal(0, sigma/np.sqrt(n_ticks), n_ticks)
    log_p_latente = np.cumsum(dW)
    # Ruído de microestrutura
    ruido = np.random.normal(0, 0.001, n_ticks)
    log_p = log_p_latente + ruido
    ret = np.diff(log_p)
    # RV clássica (viesada)
    rv_classica.append(np.sum(ret**2))
    # TSRV simplificado (two-scale)
    # escala fina e grossa
    ret_grosso = log_p[::5][1:] - log_p[::5][:-1]
    rv_fino = np.sum(ret**2)
    rv_grosso = np.sum(ret_grosso**2)
    tsrv.append(rv_grosso - (n_ticks/5)/n_ticks * rv_fino)  # aproximação

print("="*60)
print("VOLATILIDADE REALIZADA E TSRV")
print("Autor: Luiz Tiago Wilcke")
print("="*60)
print(f"RV clássica média: {np.mean(rv_classica):.6f}")
print(f"TSRV média: {np.mean(tsrv):.6f}")
print("(TSRV reduz o viés do ruído de microestrutura)")
