# ============================================================
# Modelo 19 – Econometria de Texto: EPU e LDA (tópicos)
# Fonte: Cap. 17, pp. 163–172 | EPU: pp. 164–165 | LDA: p. 165
# Autor: Luiz Tiago Wilcke
# ============================================================

import numpy as np

np.random.seed(42)

# Simular corpus de "notícias" com dicionário EPU
palavras_epu = ["incerteza", "economia", "politica", "regulacao", "deficit",
                "inflacao", "juros", "crise", "reforma", "orcamento"]
palavras_outras = ["esporte", "cultura", "tecnologia", "saude", "educacao",
                   "clima", "transporte", "energia", "comercio", "turismo"]
vocab = palavras_epu + palavras_outras
V = len(vocab)
D = 200  # documentos
# Documentos: metade com alta incerteza
docs = []
epu_true = np.zeros(D)
for d in range(D):
    if d < D // 2:
        # alta EPU
        n_words = np.random.randint(30, 60)
        w = list(np.random.choice(palavras_epu, n_words // 2)) + \
            list(np.random.choice(vocab, n_words // 2))
        epu_true[d] = 1
    else:
        n_words = np.random.randint(30, 60)
        w = list(np.random.choice(palavras_outras, 2*n_words // 3)) + \
            list(np.random.choice(vocab, n_words // 3))
        epu_true[d] = 0
    docs.append(w)

# Índice EPU lexical (contagem)
epu_score = np.array([sum(1 for w in doc if w in palavras_epu) / len(doc) for doc in docs])

# LDA simplificado (Gibbs reduzido) – 2 tópicos
K = 2
# Contagens documento-palavra
N_dv = np.zeros((D, V))
word2idx = {w: i for i, w in enumerate(vocab)}
for d, doc in enumerate(docs):
    for w in doc:
        N_dv[d, word2idx[w]] += 1

# Inicialização aleatória de tópicos
z = [np.random.randint(0, K, len(doc)) for doc in docs]
ndk = np.zeros((D, K))
nkw = np.zeros((K, V))
nk = np.zeros(K)
for d, doc in enumerate(docs):
    for i, w in enumerate(doc):
        k = z[d][i]
        ndk[d, k] += 1
        nkw[k, word2idx[w]] += 1
        nk[k] += 1

alpha, beta_dir = 0.5, 0.1
for it in range(50):
    for d, doc in enumerate(docs):
        for i, w in enumerate(doc):
            k_old = z[d][i]
            wid = word2idx[w]
            ndk[d, k_old] -= 1
            nkw[k_old, wid] -= 1
            nk[k_old] -= 1
            # p(z=k | .)
            p = (ndk[d] + alpha) * (nkw[:, wid] + beta_dir) / (nk + V * beta_dir)
            p = np.maximum(p, 0)
            p /= p.sum()
            k_new = np.random.choice(K, p=p)
            z[d][i] = k_new
            ndk[d, k_new] += 1
            nkw[k_new, wid] += 1
            nk[k_new] += 1

# Tópico dominante por documento
theta = (ndk + alpha) / (ndk.sum(axis=1, keepdims=True) + K * alpha)
topico_dom = theta.argmax(axis=1)

print("=" * 70)
print("ECONOMETRIA DE TEXTO – EPU + LDA")
print("Fonte: Cap. 17, pp. 163–172 | EPU: pp. 164–165 | LDA: p. 165")
print("Autor: Luiz Tiago Wilcke")
print("=" * 70)
print(f"Correlação EPU lexical vs indicador verdadeiro: {np.corrcoef(epu_score, epu_true)[0,1]:.3f}")
print(f"EPU médio (alta incerteza): {epu_score[:D//2].mean():.3f}")
print(f"EPU médio (baixa incerteza): {epu_score[D//2:].mean():.3f}")
print(f"Distribuição de tópicos dominantes: {np.bincount(topico_dom)}")
# Palavras top por tópico
for k in range(K):
    top_w = np.argsort(nkw[k])[::-1][:5]
    print(f"Tópico {k}: {[vocab[i] for i in top_w]}")
