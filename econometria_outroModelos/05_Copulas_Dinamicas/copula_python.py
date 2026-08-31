# ============================================================
# Modelo 05 – Cópulas e Dependência de Caudas (Clayton / Gumbel / Gaussiana)
# Fonte: Cap. 19, pp. 183–184 | Teorema de Sklar: p. 183
# Autor: Luiz Tiago Wilcke
# ============================================================

import numpy as np
from scipy.stats import norm, t as student_t
from scipy.optimize import minimize

np.random.seed(42)
n = 1000

# Simular com cópula de Clayton (dependência de cauda inferior)
theta_clayton = 2.5
u = np.random.uniform(0, 1, n)
# Método da inversão condicional para Clayton
v = np.random.uniform(0, 1, n)
w = (v**(-theta_clayton/(1+theta_clayton)) - 1) * u**(-theta_clayton) + 1
w = w**(-1/theta_clayton)
# Margens t-Student
x = student_t.ppf(u, df=5)
y = student_t.ppf(w, df=5)

# Pseudo-observações (ranks)
U = (np.argsort(np.argsort(x)) + 1) / (n + 1)
V = (np.argsort(np.argsort(y)) + 1) / (n + 1)

def clayton_loglik(theta, U, V):
    if theta <= 0:
        return 1e10
    ll = (np.log(theta + 1)
          - (theta + 1) * (np.log(U) + np.log(V))
          - (2 + 1/theta) * np.log(U**(-theta) + V**(-theta) - 1))
    return -ll.sum()

def gumbel_loglik(theta, U, V):
    if theta < 1:
        return 1e10
    t1 = (-np.log(U))**theta
    t2 = (-np.log(V))**theta
    A = (t1 + t2)**(1/theta)
    ll = (np.log(A) + (theta - 1)*np.log(np.log(U)*np.log(V))
          - np.log(U) - np.log(V)
          + np.log(t1 + t2) * (1/theta - 2)  # simplificado
          + (theta - 1) * np.log(t1/(t1+t2) * t2/(t1+t2) + 1/theta))  # approx
    # densidade Gumbel completa
    C = np.exp(-A)
    dens = (C * (U*V)**(-1) * A**(1-2*theta) * (t1*t2)**(1-1/theta)
            * (1 + (theta-1)*A**(-1)))  # forma reduzida
    dens = np.clip(dens, 1e-12, None)
    return -np.log(dens).sum()

res_c = minimize(lambda th: clayton_loglik(th[0], U, V), [1.5], method="Nelder-Mead")
# Tau de Kendall empírico
tau_emp = np.corrcoef(np.argsort(x), np.argsort(y))[0, 1]  # proxy
tau_clayton = res_c.x[0] / (res_c.x[0] + 2)

print("=" * 70)
print("CÓPULAS – DEPENDÊNCIA DE CAUDAS (Clayton)")
print("Fonte: Cap. 19, pp. 183–184 | Teorema de Sklar (1959): p. 183")
print("Autor: Luiz Tiago Wilcke")
print("=" * 70)
print(f"Theta Clayton estimado: {res_c.x[0]:.3f} (verdadeiro={theta_clayton})")
print(f"Tau de Kendall (Clayton): {tau_clayton:.3f}")
print(f"Correlação de Pearson: {np.corrcoef(x,y)[0,1]:.3f}")
