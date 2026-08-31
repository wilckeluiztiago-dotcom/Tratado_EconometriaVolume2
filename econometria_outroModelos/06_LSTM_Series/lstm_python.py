# ============================================================
# Modelo 06 – LSTM para Séries Temporais
# Fonte: Cap. 20, pp. 192–197 | Célula LSTM: pp. 193–194
# Autor: Luiz Tiago Wilcke
# ============================================================

import numpy as np

np.random.seed(42)

# --- Célula LSTM implementada do zero (pedagógico) ---
class LSTMCell:
    """
    Portas: f (forget), i (input), o (output), g (candidate)
    c_t = f_t * c_{t-1} + i_t * g_t
    h_t = o_t * tanh(c_t)
    """
    def __init__(self, input_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        scale = 0.1
        # Pesos: [x; h] -> 4 * hidden
        self.W = np.random.randn(input_dim + hidden_dim, 4 * hidden_dim) * scale
        self.b = np.zeros(4 * hidden_dim)

    def step(self, x, h_prev, c_prev):
        concat = np.hstack([x, h_prev])
        gates = concat @ self.W + self.b
        f = 1 / (1 + np.exp(-gates[:self.hidden_dim]))
        i = 1 / (1 + np.exp(-gates[self.hidden_dim:2*self.hidden_dim]))
        o = 1 / (1 + np.exp(-gates[2*self.hidden_dim:3*self.hidden_dim]))
        g = np.tanh(gates[3*self.hidden_dim:])
        c = f * c_prev + i * g
        h = o * np.tanh(c)
        return h, c

# Série AR com não-linearidade
T = 400
y = np.zeros(T)
for t in range(1, T):
    y[t] = 0.7*y[t-1] - 0.2*y[t-1]**3 + np.random.normal(0, 0.3)

# Treinar LSTM simples (1 passo à frente) por GD
hidden = 8
cell = LSTMCell(1, hidden)
W_out = np.random.randn(hidden) * 0.1
b_out = 0.0
lr = 0.01
seq_len = 10
losses = []

for epoch in range(60):
    loss_epoch = 0.0
    n_batch = 0
    for start in range(0, T - seq_len - 1, seq_len):
        h = np.zeros(hidden)
        c = np.zeros(hidden)
        for t in range(start, start + seq_len):
            h, c = cell.step(np.array([y[t]]), h, c)
        pred = h @ W_out + b_out
        target = y[start + seq_len]
        err = pred - target
        loss_epoch += err**2
        n_batch += 1
        # gradiente simplificado só em W_out (pedagógico)
        W_out -= lr * err * h
        b_out -= lr * err
    losses.append(loss_epoch / max(n_batch, 1))

# Previsão
h = np.zeros(hidden); c = np.zeros(hidden)
preds = []
for t in range(T - 50, T):
    h, c = cell.step(np.array([y[t-1] if t > 0 else 0]), h, c)
    preds.append(h @ W_out + b_out)
preds = np.array(preds)
rmse = np.sqrt(np.mean((preds - y[-50:])**2))

print("=" * 70)
print("LSTM – LONG SHORT-TERM MEMORY (implementação do zero)")
print("Fonte: Cap. 20, pp. 192–197 | Célula LSTM: pp. 193–194")
print("Autor: Luiz Tiago Wilcke")
print("=" * 70)
print(f"Loss final (MSE treino): {losses[-1]:.4f}")
print(f"RMSE previsão (últimos 50): {rmse:.4f}")
print(f"Desvio-padrão da série: {y.std():.4f}")
