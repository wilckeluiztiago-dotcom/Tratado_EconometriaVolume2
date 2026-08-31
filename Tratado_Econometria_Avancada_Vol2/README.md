# Tratado de Econometria Avançada — Volume II

**Teoria, Modelagem Matemática e Implementação com Python e R**

**Autor:** Luiz Tiago Wilcke

---

## Descrição

Este repositório contém **20 modelos econométricos completos** baseados no livro *Tratado de Econometria Avançada – Volume II*. Cada modelo inclui:

- Formulação teórica e equações
- Geração de dados sintéticos
- Implementação completa em **Python**
- Implementação completa em **R**
- Estimação, diagnóstico e visualização

Os códigos são didáticos, autocontidos e prontos para execução.

---

## Estrutura do Projeto

```
Tratado_Econometria_Avancada_Vol2/
├── README.md
├── 01_Filtro_Kalman/
├── 02_Controle_Sintetico/
├── 03_Double_ML/
├── 04_GMM/
├── 05_TAR/
├── 06_STAR/
├── 07_Markov_Switching/
├── 08_DSGE/
├── 09_Lasso_Ridge/
├── 10_Causal_Forest/
├── 11_SAR_Espacial/
├── 12_Painel_Fatores/
├── 13_Regressao_Quantilica/
├── 14_Bayesiano_MCMC/
├── 15_Efeitos_Pares/
├── 16_Volatilidade_Realizada/
├── 17_Logit_Misto/
├── 18_VAR_VEC/
├── 19_Arellano_Bond/
└── 20_RDD_Sharp/
```

Cada pasta contém um script Python e um script R independentes.

---

## Requisitos

### Python
```bash
pip install numpy pandas scipy statsmodels scikit-learn matplotlib seaborn
pip install linearmodels arch cvxpy networkx
```

### R
```r
install.packages(c("dlm", "Synth", "gmm", "tsDyn", "vars", "plm",
                   "quantreg", "spatialreg", "spdep", "glmnet",
                   "randomForest", "urca", "AER", "ggplot2"))
```

---

# Modelos: Teoria, Equações e Explicação Detalhada

---

## 01 — Filtro de Kalman (Modelo de Nível Local)

**Referência:** Capítulo 1 — Modelagem de Espaço de Estados e o Filtro de Kalman

### Ideia central

O Filtro de Kalman estima de forma recursiva o estado latente de um sistema dinâmico linear a partir de observações ruidosas. Separar a dinâmica não observada (estado) da equação de medida permite filtrar tendência, produto potencial, taxa de juros natural e outras variáveis latentes comuns em macroeconomia.

### Formulação em espaço de estados

**Equação de transição (estado latente):**

$$
\alpha_t = T_t \alpha_{t-1} + R_t \eta_t, \qquad \eta_t \sim N(0, Q_t)
$$

**Equação de medida (observação):**

$$
y_t = Z_t \alpha_t + d_t + \epsilon_t, \qquad \epsilon_t \sim N(0, H_t)
$$

com $\mathrm{Cov}(\eta_t, \epsilon_s) = 0$ para todo $t, s$.

### Modelo de nível local (caso univariado)

$$
\begin{aligned}
y_t &= \alpha_t + \epsilon_t, & \epsilon_t &\sim N(0, H) \\
\alpha_t &= \alpha_{t-1} + \eta_t, & \eta_t &\sim N(0, Q)
\end{aligned}
$$

### Equações recursivas do filtro

**Predição (a priori):**

$$
\begin{aligned}
a_{t|t-1} &= T_t a_{t-1} \\
P_{t|t-1} &= T_t P_{t-1} T_t^\top + R_t Q_t R_t^\top
\end{aligned}
$$

**Inovação e variância da inovação:**

$$
\begin{aligned}
v_t &= y_t - Z_t a_{t|t-1} - d_t \\
F_t &= Z_t P_{t|t-1} Z_t^\top + H_t
\end{aligned}
$$

**Ganho de Kalman e atualização (a posteriori):**

$$
\begin{aligned}
K_t &= P_{t|t-1} Z_t^\top F_t^{-1} \\
a_t &= a_{t|t-1} + K_t v_t \\
P_t &= P_{t|t-1} - K_t Z_t P_{t|t-1}
\end{aligned}
$$

Forma de Joseph (numericamente estável):

$$
P_t = (I - K_t Z_t) P_{t|t-1} (I - K_t Z_t)^\top + K_t H_t K_t^\top
$$

### Interpretação

O ganho $K_t$ pondera a inovação $v_t$: quando a incerteza a priori $P_{t|t-1}$ é grande em relação ao ruído de medida $H_t$, o filtro confia mais na nova observação. O algoritmo é ótimo no sentido de mínimo erro quadrático médio entre os estimadores lineares, mesmo sem gaussianidade (sob ortogonalidade e variâncias finitas).

### Arquivos
- `01_Filtro_Kalman/kalman_python.py`
- `01_Filtro_Kalman/kalman_r.R`

---

## 02 — Controle Sintético (Synthetic Control)

**Referência:** Capítulo 2 — Controle Composto e Inferência Causal Avançada

### Ideia central

Quando há uma única unidade tratada (país, estado, município) e um conjunto de unidades não tratadas (pool de doadores), o método constrói um controle artificial como combinação linear convexa das doadoras, de modo a reproduzir o melhor possível a trajetória pré-tratamento da unidade tratada.

### Resultados potenciais

$$
Y_{it} = Y_{it}^{N} + \tau_{it} D_{it}
$$

onde $D_{it}=1$ se a unidade $i$ está tratada no tempo $t$, e $\tau_{it}$ é o efeito do tratamento.

### Problema de otimização dos pesos

Seja $Y_1^{\mathrm{pre}}$ o vetor de resultados pré-tratamento da unidade tratada e $Y_0^{\mathrm{pre}}$ a matriz $J \times T_0$ das $J$ doadoras. Os pesos $\mathbf{w}^* = (w_2,\ldots,w_{J+1})$ resolvem:

$$
\mathbf{w}^* = \arg\min_{\mathbf{w}} \left\| Y_1^{\mathrm{pre}} - Y_0^{\mathrm{pre}\top} \mathbf{w} \right\|_V^2
$$

sujeito a:

$$
w_j \ge 0, \qquad \sum_{j=2}^{J+1} w_j = 1
$$

A matriz de ponderação $V$ (frequentemente diagonal) privilegia preditores pré-tratamento relevantes.

### Controle sintético e efeito

$$
\hat{Y}_{1t}^{N} = \sum_{j=2}^{J+1} w_j^* Y_{jt}, \qquad
\hat{\tau}_{1t} = Y_{1t} - \hat{Y}_{1t}^{N}, \quad t > T_0
$$

### Interpretação

O método evita extrapolar para fora do suporte convexo das doadoras e produz um contrafactual transparente. Inferência clássica baseia-se em testes de permutação (placebo) sobre o pool de doadores.

### Arquivos
- `02_Controle_Sintetico/controle_sintetico_python.py`
- `02_Controle_Sintetico/controle_sintetico_r.R`

---

## 03 — Double Machine Learning (DML)

**Referência:** Capítulo 3 — Double Machine Learning e Causal Forests

### Ideia central

Em modelos parcialmente lineares com nuisance functions de alta dimensão, regularização (Lasso, florestas, boosting) introduz viés de regularização. O DML usa ortogonalização de Neyman e cross-fitting para obter estimativas $\sqrt{n}$-consistentes e assintoticamente normais do parâmetro de interesse.

### Modelo parcialmente linear

$$
\begin{aligned}
Y &= \theta_0 D + g_0(X) + U, & \mathbb{E}[U \mid X, D] &= 0 \\
D &= m_0(X) + V, & \mathbb{E}[V \mid X] &= 0
\end{aligned}
$$

### Score de Neyman (ortogonal)

$$
\psi(W; \theta, \eta) = \big(Y - g(X) - \theta(D - m(X))\big)\,(D - m(X))
$$

onde $\eta = (g, m)$. A derivada de Gateaux em relação a $\eta$ anula-se em $\eta_0$, o que torna o estimador de $\theta$ insensível a erros de primeira ordem na estimação das nuisance functions.

### Algoritmo com cross-fitting

1. Particionar a amostra em $K$ folds.
2. Em cada fold $k$, estimar $\hat{g}^{(-k)}$ e $\hat{m}^{(-k)}$ nos dados fora de $k$.
3. Formar resíduos ortogonalizados no fold $k$:
   $$
   \hat{U}_i = Y_i - \hat{g}^{(-k)}(X_i), \qquad
   \hat{V}_i = D_i - \hat{m}^{(-k)}(X_i)
   $$
4. Estimar $\theta$ por regressão residual empilhada:
   $$
   \hat{\theta} = \left(\sum_i \hat{V}_i^2\right)^{-1} \sum_i \hat{V}_i \hat{U}_i
   $$

### Interpretação

O cross-fitting evita overfitting na mesma amostra usada para estimar $\theta$. O estimador resultante é double robust: basta que uma das nuisance functions seja consistentemente estimada (sob taxas adequadas) para $\hat{\theta}$ ser consistente.

### Arquivos
- `03_Double_ML/dml_python.py`
- `03_Double_ML/dml_r.R`

---

## 04 — GMM e Identificação Fraca

**Referência:** Capítulo 4 — GMM Avançado e Identificação Fraca

### Ideia central

O Método Generalizado dos Momentos estima parâmetros a partir de condições de ortogonalidade $\mathbb{E}[g(W_i, \theta_0)] = 0$. Em presença de instrumentos fracos, a distribuição assintótica usual de Wald quebra; métodos robustos (Anderson-Rubin, Kleibergen) são necessários.

### Estimador GMM

$$
\hat{\theta}_{\mathrm{GMM}} = \arg\min_{\theta} \, n \, \bar{g}(\theta)^\top W \bar{g}(\theta)
$$

onde $\bar{g}(\theta) = n^{-1}\sum_i g(W_i, \theta)$ e $W$ é matriz de ponderação positiva definida.

### Eficiência de duas etapas

1. Obter $\hat{\theta}_1$ com $W = I$.
2. Estimar $S = \mathrm{Var}(g(W_i, \theta_0))$ em $\hat{\theta}_1$.
3. Reestimar com $W = \hat{S}^{-1}$ (ponderação ótima).

O estimador de atualização contínua (CUE) minimiza o critério com $W(\theta) = S(\theta)^{-1}$ atualizado a cada avaliação, com melhores propriedades sob identificação fraca.

### Momentos em regressão instrumental

$$
g(W_i, \theta) = Z_i \big(Y_i - X_i^\top \theta\big)
$$

### Teste de Anderson-Rubin (robusto a instrumentos fracos)

Sob $H_0: \theta = \theta_0$,

$$
AR(\theta_0) = \frac{n}{2} \bar{g}(\theta_0)^\top \hat{S}^{-1} \bar{g}(\theta_0)
$$

tem distribuição $\chi^2$ livre de parâmetros de nuisance sob identificação fraca.

### Arquivos
- `04_GMM/gmm_python.py`
- `04_GMM/gmm_r.R`

---

## 05 — Modelo TAR (Threshold Autoregressive)

**Referência:** Capítulo 5 — Séries Temporais Não-Lineares e Regime Switching

### Ideia central

O TAR permite que a dinâmica autoregressiva mude de regime de forma abrupta quando uma variável de limiar (tipicamente o próprio lag) cruza um valor crítico.

### Especificação TAR(1) de dois regimes

$$
y_t =
\begin{cases}
\phi_{10} + \phi_{11} y_{t-1} + \varepsilon_t, & y_{t-1} \le c \\
\phi_{20} + \phi_{21} y_{t-1} + \varepsilon_t, & y_{t-1} > c
\end{cases}
$$

com $\varepsilon_t \sim \mathrm{iid}(0, \sigma^2)$.

### Estimação

Para $c$ fixo, os coeficientes de cada regime são estimados por MQO separado. O limiar $\hat{c}$ é obtido por busca em grade (ou otimização) minimizando a soma de quadrados residual:

$$
\hat{c} = \arg\min_c \, \mathrm{SSR}(c)
$$

### Interpretação

Útil para assimetrias de ciclo de negócios, zonas de não-arbitragem em finanças e mudanças estruturais endógenas. A transição é descontínua no limiar.

### Arquivos
- `05_TAR/tar_python.py`
- `05_TAR/tar_r.R`

---

## 06 — Modelo STAR (Smooth Transition Autoregressive)

**Referência:** Capítulo 5 — Séries Temporais Não-Lineares

### Ideia central

O STAR generaliza o TAR substituindo a mudança abrupta por uma função de transição suave, permitindo regimes intermediários.

### LSTAR (transição logística)

$$
y_t = \big(\phi_{10} + \phi_{11} y_{t-1}\big)\big(1 - G(y_{t-1}; \gamma, c)\big)
+ \big(\phi_{20} + \phi_{21} y_{t-1}\big) G(y_{t-1}; \gamma, c) + \varepsilon_t
$$

$$
G(s; \gamma, c) = \frac{1}{1 + \exp\big(-\gamma (s - c)\big)}, \qquad \gamma > 0
$$

### ESTAR (transição exponencial)

$$
G(s; \gamma, c) = 1 - \exp\big(-\gamma (s - c)^2\big)
$$

### Interpretação

O parâmetro $\gamma$ controla a velocidade da transição: $\gamma \to \infty$ recupera o TAR; $\gamma \to 0$ colapsa para um AR linear. O ESTAR é simétrico em torno de $c$ e útil para modelar correção de desvios em ambas as direções (ex.: paridade do poder de compra).

### Arquivos
- `06_STAR/star_python.py`
- `06_STAR/star_r.R`

---

## 07 — Markov Switching (Filtro de Hamilton)

**Referência:** Capítulo 5 — Regime Switching

### Ideia central

Os regimes são governados por uma cadeia de Markov oculta. O Filtro de Hamilton atualiza de forma recursiva as probabilidades filtradas de cada regime.

### Modelo de média com dois regimes

$$
y_t = \mu_{S_t} + \varepsilon_t, \qquad \varepsilon_t \sim N(0, \sigma_{S_t}^2)
$$

$$
P(S_t = j \mid S_{t-1} = i) = p_{ij}, \qquad \sum_j p_{ij} = 1
$$

### Filtro de Hamilton (resumo)

1. **Predição das probabilidades:**
   $$
   \xi_{t|t-1} = P^\top \xi_{t-1|t-1}
   $$
2. **Densidade condicional ao regime** e atualização de Bayes:
   $$
   \xi_{t|t} \propto \xi_{t|t-1} \odot \eta_t
   $$
   onde $\eta_t$ é o vetor de densidades $f(y_t \mid S_t = j)$.

A verossimilhança é o produto das densidades preditivas marginais. Estimação tipicamente por EM ou maximização numérica direta.

### Interpretação

Captura mudanças de regime persistentes (expansão/recessão, alta/baixa volatilidade) sem exigir observação direta do estado. As probabilidades suavizadas $\xi_{t|T}$ usam informação de toda a amostra.

### Arquivos
- `07_Markov_Switching/ms_python.py`
- `07_Markov_Switching/ms_r.R`

---

## 08 — DSGE Novo-Keynesiano (Três Equações)

**Referência:** Capítulo 6 — Modelos DSGE

### Ideia central

O modelo canônico de três equações descreve a dinâmica do hiato do produto, da inflação e da taxa de juros sob expectativas racionais e rigidez nominal.

### Sistema log-linearizado

**Curva IS dinâmica (Euler de consumo):**

$$
y_t = \mathbb{E}_t y_{t+1} - \sigma^{-1}\big(i_t - \mathbb{E}_t \pi_{t+1} - r_t^n\big)
$$

**Curva de Phillips Novo-Keynesiana (NKPC):**

$$
\pi_t = \beta \mathbb{E}_t \pi_{t+1} + \kappa y_t + u_t
$$

**Regra de Taylor:**

$$
i_t = \phi_\pi \pi_t + \phi_y y_t + v_t
$$

### Solução de Blanchard-Kahn

O sistema é escrito em forma de expectativas racionais:

$$
\mathbb{E}_t X_{t+1} = A X_t + B \varepsilon_t
$$

A existência e unicidade de solução estacionária requerem que o número de autovalores de $A$ fora do círculo unitário coincida com o número de variáveis forward-looking.

### Interpretação

Parâmetros estruturais ($\sigma$, $\kappa$, $\phi_\pi$, $\phi_y$) têm interpretação microfundamentada. Estimação bayesiana combina o filtro de Kalman (para estados latentes e verossimilhança) com MCMC sobre a posterior dos parâmetros.

### Arquivos
- `08_DSGE/dsge_python.py`
- `08_DSGE/dsge_r.R`

---

## 09 — Regularização Linear: Ridge, Lasso e Elastic Net

**Referência:** Capítulo 7 — Machine Learning Aplicado a Macro e Microeconometria

### Ideia central

Em regressão de alta dimensão ($p$ grande em relação a $n$), MQO é instável ou indefinido. Regularização penaliza a complexidade dos coeficientes e induz estabilidade (e esparsidade no caso L1).

### Ridge (penalidade $L_2$)

$$
\hat{\beta}^{\mathrm{Ridge}} = \arg\min_\beta \, \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2
$$

Solução fechada: $\hat{\beta}^{\mathrm{Ridge}} = (X^\top X + \lambda I)^{-1} X^\top y$.

### Lasso (penalidade $L_1$)

$$
\hat{\beta}^{\mathrm{Lasso}} = \arg\min_\beta \, \|y - X\beta\|_2^2 + \lambda \|\beta\|_1
$$

Induz esparsidade: muitos coeficientes exatos iguais a zero (seleção de variáveis).

### Elastic Net

$$
\hat{\beta}^{\mathrm{EN}} = \arg\min_\beta \, \|y - X\beta\|_2^2 + \lambda\big(\alpha \|\beta\|_1 + (1-\alpha)\|\beta\|_2^2\big)
$$

Combina seleção ($L_1$) com agrupamento de coeficientes correlacionados ($L_2$).

### Interpretação

O parâmetro $\lambda$ é tipicamente escolhido por validação cruzada. Em econometria, regularização é usada tanto para previsão quanto como primeira etapa de procedimentos de inferência causal de alta dimensão (por exemplo, DML).

### Arquivos
- `09_Lasso_Ridge/regularizacao_python.py`
- `09_Lasso_Ridge/regularizacao_r.R`

---

## 10 — Causal Forest (Floresta Causal)

**Referência:** Capítulos 3 e 7

### Ideia central

Florestas causais estimam efeitos de tratamento heterogêneos (CATE) de forma não-paramétrica, com validação honest (honesty) e inferência assintótica para o efeito local.

### CATE

$$
\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]
$$

### Honestidade

A amostra é dividida em:
- subamostra de **estrutura** (definição dos nós da árvore);
- subamostra de **estimação** (cálculo das médias de tratamento/controle dentro de cada folha).

Isso evita viés de overfitting na estimativa do CATE.

### Estimador em cada folha

$$
\hat{\tau}_L = \bar{Y}_{L,1} - \bar{Y}_{L,0}
$$

A floresta agrega muitas árvores honestas. Versões modernas usam kernels adaptativos centrados e ponderação local para obter normalidade assintótica pontual.

### Interpretação

Permite mapear para quem o tratamento funciona mais (ou menos), sem impor forma funcional global. Complementa o DML, que foca no ATE (efeito médio).

### Arquivos
- `10_Causal_Forest/causal_forest_python.py`
- `10_Causal_Forest/causal_forest_r.R`

---

## 11 — Modelo SAR (Spatial Autoregressive)

**Referência:** Capítulo 8 — Econometria Espacial Avançada

### Ideia central

O SAR incorpora dependência espacial no resultado: o valor de $y$ em uma região depende dos valores de $y$ nas regiões vizinhas, além de covariáveis locais.

### Especificação

$$
y = \rho W y + X\beta + \varepsilon, \qquad \varepsilon \sim N(0, \sigma^2 I)
$$

onde $W$ é a matriz de pesos espaciais (contigüidade, $k$-vizinhos, distância inversa etc.), tipicamente normalizada por linha.

### Forma reduzida

$$
y = (I - \rho W)^{-1} X\beta + (I - \rho W)^{-1}\varepsilon
$$

### Estimação

MQO é inconsistente porque $Wy$ é endógeno. Métodos usuais:
- Máxima verossimilhança;
- 2SLS / GMM com instrumentos $WX$, $W^2 X$, etc. (Kelejian-Prucha).

### Interpretação

O parâmetro $\rho$ mede a força do spillover espacial. Efeitos totais, diretos e indiretos são obtidos a partir de $(I - \rho W)^{-1}$.

### Arquivos
- `11_SAR_Espacial/sar_python.py`
- `11_SAR_Espacial/sar_r.R`

---

## 12 — Painel com Fatores Comuns (CCE de Pesaran)

**Referência:** Capítulo 9 — Painéis com $N$ e $T$ Grandes

### Ideia central

Quando erros (e regressores) compartilham fatores comuns não observados, estimadores de efeitos fixos convencionais são inconsistentes. O estimador CCE usa médias transversais como proxies dos fatores.

### Modelo de fator interativo

$$
y_{it} = \beta^\top x_{it} + \lambda_i^\top f_t + \varepsilon_{it}
$$

### Estimador CCE

Para cada unidade $i$, estimar por MQO:

$$
y_{it} = \beta^\top x_{it} + \gamma_{i1}^\top \bar{y}_t + \gamma_{i2}^\top \bar{x}_t + u_{it}
$$

onde $\bar{y}_t = N^{-1}\sum_i y_{it}$ e $\bar{x}_t = N^{-1}\sum_i x_{it}$. O estimador CCEMG é a média dos $\hat{\beta}_i$.

### Interpretação

Sob rank condition adequada, as médias transversais geram o espaço dos fatores comuns, eliminando a dependência transversal. Alternativas incluem o estimador de componentes principais iterativos (Bai) e testes de dependência transversal (CD de Pesaran).

### Arquivos
- `12_Painel_Fatores/cce_python.py`
- `12_Painel_Fatores/cce_r.R`

---

## 13 — Regressão Quantílica

**Referência:** Capítulo 10 — Regressão Quantílica e Métodos Distribucionais

### Ideia central

Em vez de modelar apenas a média condicional, a regressão quantílica estima quantis condicionais $\mathbb{Q}_\tau(Y \mid X)$, revelando heterogeneidade ao longo da distribuição.

### Função de perda de pinball (check function)

$$
\rho_\tau(u) = u\big(\tau - \mathbf{1}\{u < 0\}\big)
$$

### Estimador

$$
\hat{\beta}(\tau) = \arg\min_\beta \sum_{i=1}^n \rho_\tau\big(Y_i - X_i^\top \beta\big)
$$

### Propriedades

Sob condições de regularidade, $\sqrt{n}(\hat{\beta}(\tau) - \beta(\tau))$ é assintoticamente normal. Processos de coeficientes $\tau \mapsto \hat{\beta}(\tau)$ permitem testar igualdade de slopes entre quantis e homogeneidade de efeitos.

### Interpretação

Essencial quando o interesse está em caudas (pobreza, retornos extremos, salários no topo) ou quando há heterocedasticidade que altera a forma da distribuição condicional.

### Arquivos
- `13_Regressao_Quantilica/quantilica_python.py`
- `13_Regressao_Quantilica/quantilica_r.R`

---

## 14 — Econometria Bayesiana (MCMC / Gibbs)

**Referência:** Capítulo 11 — Econometria Bayesiana

### Ideia central

A inferência bayesiana combina a verossimilhança com uma prior e resume a incerteza via a distribuição a posteriori. Quando a posterior não é analítica, MCMC (Gibbs, Metropolis-Hastings) gera amostras da posterior.

### Teorema de Bayes

$$
p(\theta \mid y) = \frac{p(y \mid \theta)\, p(\theta)}{p(y)} \propto p(y \mid \theta)\, p(\theta)
$$

### Gibbs Sampler (regressão linear normal)

Prior conjugada: $\beta \mid \sigma^2 \sim N(b_0, V_0)$, $\sigma^2 \sim \mathrm{IG}(a_0, b_0)$.

Iterar:

1. Amostrar $\beta^{(s)} \mid \sigma^{2(s-1)}, y \sim N(m_n, V_n)$;
2. Amostrar $\sigma^{2(s)} \mid \beta^{(s)}, y \sim \mathrm{IG}(a_n, b_n)$.

Após burn-in, a cadeia $\{\theta^{(s)}\}$ aproxima a posterior conjunta.

### Interpretação

Intervalos credíveis têm interpretação probabilística direta. Modelos hierárquicos e com parâmetros latentes (espaço de estados, mistura) encaixam-se naturalmente no arcabouço de Gibbs / data augmentation.

### Arquivos
- `14_Bayesiano_MCMC/mcmc_python.py`
- `14_Bayesiano_MCMC/mcmc_r.R`

---

## 15 — Efeitos de Pares em Redes (Linear-in-Means)

**Referência:** Capítulo 12 — Econometria de Redes

### Ideia central

O resultado de um indivíduo depende da média dos resultados (e das características) de seus pares na rede. O problema de reflexão de Manski impede a identificação separada de efeitos endógenos, contextuais e correlacionados sob especificação linear-in-means sem estrutura adicional.

### Modelo de Bramoullé, Djebbari e Fortin (2009)

$$
y = \alpha\iota + \beta G y + \gamma X + \delta G X + \varepsilon
$$

onde $G$ é a matriz de adjacência normalizada por linha (médias dos pares).

### Identificação

Se $I$, $G$ e $G^2$ são linearmente independentes, os parâmetros $(\beta, \gamma, \delta)$ são identificados. Instrumentos naturais para $Gy$ incluem $G^2 X$ (características dos pares dos pares).

### Interpretação

$\beta$ captura o efeito endógeno (contágio social); $\delta$ o efeito contextual. Aplicações: educação, crime, finanças, redes de firmas.

### Arquivos
- `15_Efeitos_Pares/network_python.py`
- `15_Efeitos_Pares/network_r.R`

---

## 16 — Volatilidade Realizada e Estimadores Robustos ao Ruído

**Referência:** Capítulo 13 — Econometria de Alta Frequência

### Ideia central

A variância integrada de um processo de semimartingale pode ser estimada pela soma de retornos ao quadrado de alta frequência (volatilidade realizada). Ruído de microestrutura e saltos enviesam o estimador clássico; TSRV, bipower variation e realized kernels corrigem esses problemas.

### Volatilidade realizada clássica

$$
RV_t = \sum_{i=1}^{n} r_{t,i}^2
$$

Sob amostragem cada vez mais frequente e ausência de ruído, $RV_t \xrightarrow{p} \int_{t-1}^t \sigma_s^2 \, ds$.

### Two-Scale Realized Volatility (TSRV)

Combina uma escala fina (viesada pelo ruído) e uma escala grossa:

$$
\widehat{\mathrm{TSRV}} = \widehat{RV}^{\mathrm{grosso}} - \frac{\bar{n}}{n} \widehat{RV}^{\mathrm{fino}}
$$

(ajustes adicionais de viés e de graus de liberdade são usados na prática).

### Interpretação

Essencial para precificação de opções, gestão de risco e modelagem de volatilidade estocástica com dados intradiários.

### Arquivos
- `16_Volatilidade_Realizada/vol_python.py`
- `16_Volatilidade_Realizada/vol_r.R`

---

## 17 — Logit Misto (Mixed Logit)

**Referência:** Capítulo 14 — Modelos de Escolha Discreta e BLP

### Ideia central

O logit multinomial sofre da propriedade IIA. O logit misto permite coeficientes aleatórios (heterogeneidade não observada), gerando padrões de substituição flexíveis.

### Utilidade

$$
U_{ij} = x_{ij}^\top \beta_i + \varepsilon_{ij}, \qquad \varepsilon_{ij} \sim \mathrm{Gumbel}
$$

$$
\beta_i \sim f(\beta \mid \theta)
$$

### Probabilidade de escolha (simulada)

$$
P_{ij}(\theta) = \int \frac{\exp(x_{ij}^\top \beta)}{\sum_k \exp(x_{ik}^\top \beta)} f(\beta \mid \theta) \, d\beta
$$

A integral é aproximada por simulação de Monte Carlo (MSLE — maximum simulated likelihood).

### Interpretação

Base para estimação de demanda estrutural (BLP) quando combinado com inversão de shares de mercado e instrumentos para preços endógenos.

### Arquivos
- `17_Logit_Misto/logit_misto_python.py`
- `17_Logit_Misto/logit_misto_r.R`

---

## 18 — VAR, Cointegração e VEC

**Referência:** Capítulo 24 — Modelos VAR, Cointegração e Vetor de Correção de Erros

### Ideia central

Sistemas de séries $I(1)$ podem ser cointegrados: combinações lineares estacionárias existem mesmo que as séries individuais não o sejam. O VECM representa a dinâmica de curto prazo e a correção em direção ao equilíbrio de longo prazo.

### VAR em níveis

$$
Y_t = A_1 Y_{t-1} + \cdots + A_p Y_{t-p} + \varepsilon_t
$$

### Representação VECM (teorema de Granger)

$$
\Delta Y_t = \Pi Y_{t-1} + \sum_{j=1}^{p-1} \Gamma_j \Delta Y_{t-j} + \varepsilon_t
$$

$$
\Pi = \alpha \beta^\top
$$

onde $\beta$ contém os vetores de cointegração e $\alpha$ as velocidades de ajustamento.

### Teste de Johansen

Decomposição de posto de $\Pi$ via autovalores da equação generalizada de autovalores; estatísticas do traço e do máximo autovalor testam o número de relações de cointegração.

### Interpretação

Fundamental em macroeconomia empírica (moeda, produto, preços; paridades; demanda por moeda) e em finanças (pares cointegrados).

### Arquivos
- `18_VAR_VEC/var_python.py`
- `18_VAR_VEC/var_r.R`

---

## 19 — Arellano-Bond (GMM em Diferenças) e Blundell-Bond

**Referência:** Capítulo 25 — Dados de Painel Dinâmicos

### Ideia central

Em painéis dinâmicos com efeitos fixos, o estimador de efeitos fixos (within) é viesado para $T$ finito (viés de Nickell). Arellano-Bond usa condições de momento em diferenças com lags em níveis como instrumentos.

### Modelo

$$
y_{it} = \rho y_{i,t-1} + \beta^\top x_{it} + \alpha_i + \varepsilon_{it}
$$

### Diferenças e momentos (Arellano-Bond)

$$
\Delta y_{it} = \rho \Delta y_{i,t-1} + \beta^\top \Delta x_{it} + \Delta \varepsilon_{it}
$$

$$
\mathbb{E}\big[y_{i,t-s} \Delta \varepsilon_{it}\big] = 0, \qquad s \ge 2
$$

### Blundell-Bond (system GMM)

Sob estacionariedade inicial, adiciona-se momentos em níveis com lags em diferenças como instrumentos, melhorando a precisão quando $\rho$ é próximo de 1.

### Interpretação

Padrão-ouro para painéis micro com $N$ grande e $T$ pequeno (crescimento, investimento, demanda de trabalho). Diagnósticos: teste de Hansen/Sargan e teste de autocorrelação de Arellano-Bond.

### Arquivos
- `19_Arellano_Bond/arellano_bond_python.py`
- `19_Arellano_Bond/arellano_bond_r.R`

---

## 20 — Regressão de Descontinuidade (Sharp RDD)

**Referência:** Capítulo 30 — Regressão de Descontinuidade: Sharp e Fuzzy RDD

### Ideia central

Quando o tratamento é atribuído deterministicamente a partir de um limiar de uma running variable, o salto no resultado no limiar identifica o efeito causal local sob continuidade do potencial de resultado.

### Sharp RDD

$$
D_i = \mathbf{1}\{X_i \ge c\}
$$

$$
\tau = \lim_{x \downarrow c} \mathbb{E}[Y \mid X = x] - \lim_{x \uparrow c} \mathbb{E}[Y \mid X = x]
$$

### Estimação por polinômio local

Em uma janela $[c-h, c+h]$, estimar regressões locais à esquerda e à direita do limiar; a diferença das previsões em $c$ é $\hat{\tau}$. Bandwidth $h$ escolhido por métodos plug-in ou validação cruzada (Imbens-Kalyanaraman, Calonico-Cattaneo-Titiunik).

### Validação

- Teste de densidade de McCrary (manipulação da running variable);
- Equilíbrio de covariáveis pré-tratamento no limiar;
- Placebos em falsos limiares.

### Interpretação

Um dos desenhos quasi-experimentais mais críveis quando o limiar é exógeno e não há manipulação. O efeito é local (LATE no limiar).

### Arquivos
- `20_RDD_Sharp/rdd_python.py`
- `20_RDD_Sharp/rdd_r.R`

---

## Como Executar

```bash
# Python
python 01_Filtro_Kalman/kalman_python.py

# R
Rscript 01_Filtro_Kalman/kalman_r.R
```

---

## Observações Metodológicas

- Os códigos priorizam clareza didática e replicabilidade.
- Os resultados são ilustrativos; em aplicações reais recomenda-se validação com dados observados e diagnóstico completo.
- O projeto segue a filosofia do livro: **teoria + modelagem matemática + implementação**.

---

## Citação

Wilcke, Luiz Tiago. *Tratado de Econometria Avançada – Volume II: Teoria, Modelagem Matemática e Implementação com Python e R*. 2024.

---

**Autor:** Luiz Tiago Wilcke  
**Ano:** 2024–2026  
**Licença de uso didático:** Livre para fins educacionais e de pesquisa, com atribuição.
