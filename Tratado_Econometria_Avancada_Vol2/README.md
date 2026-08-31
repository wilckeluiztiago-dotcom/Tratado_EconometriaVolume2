# Tratado de Econometria Avançada — Volume II

**Teoria, Modelagem Matemática e Implementação com Python e R**

**Autor:** Luiz Tiago Wilcke

---

## Descrição

Este repositório contém **20 modelos econométricos completos** baseados no livro *Tratado de Econometria Avançada – Volume II*. Cada modelo inclui:

- Formulação teórica resumida
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
│   ├── kalman_python.py
│   └── kalman_r.R
├── 02_Controle_Sintetico/
│   ├── controle_sintetico_python.py
│   └── controle_sintetico_r.R
├── 03_Double_ML/
│   ├── dml_python.py
│   └── dml_r.R
├── 04_GMM/
│   ├── gmm_python.py
│   └── gmm_r.R
├── 05_TAR/
│   ├── tar_python.py
│   └── tar_r.R
├── 06_STAR/
│   ├── star_python.py
│   └── star_r.R
├── 07_Markov_Switching/
│   ├── ms_python.py
│   └── ms_r.R
├── 08_DSGE/
│   ├── dsge_python.py
│   └── dsge_r.R
├── 09_Lasso_Ridge/
│   ├── regularizacao_python.py
│   └── regularizacao_r.R
├── 10_Causal_Forest/
│   ├── causal_forest_python.py
│   └── causal_forest_r.R
├── 11_SAR_Espacial/
│   ├── sar_python.py
│   └── sar_r.R
├── 12_Painel_Fatores/
│   ├── cce_python.py
│   └── cce_r.R
├── 13_Regressao_Quantilica/
│   ├── quantilica_python.py
│   └── quantilica_r.R
├── 14_Bayesiano_MCMC/
│   ├── mcmc_python.py
│   └── mcmc_r.R
├── 15_Efeitos_Pares/
│   ├── network_python.py
│   └── network_r.R
├── 16_Volatilidade_Realizada/
│   ├── vol_python.py
│   └── vol_r.R
├── 17_Logit_Misto/
│   ├── logit_misto_python.py
│   └── logit_misto_r.R
├── 18_VAR_VEC/
│   ├── var_python.py
│   └── var_r.R
├── 19_Arellano_Bond/
│   ├── arellano_bond_python.py
│   └── arellano_bond_r.R
└── 20_RDD_Sharp/
    ├── rdd_python.py
    └── rdd_r.R
```

---

## Lista dos 20 Modelos

| Nº | Modelo | Capítulo de Referência |
|----|--------|------------------------|
| 01 | Filtro de Kalman (Nível Local) | Cap. 1 – Espaço de Estados |
| 02 | Controle Sintético (Synthetic Control) | Cap. 2 – Controle Composto |
| 03 | Double Machine Learning (DML) | Cap. 3 – DML e Causal Forests |
| 04 | GMM e Identificação Fraca | Cap. 4 – GMM Avançado |
| 05 | Modelo TAR (Threshold Autoregressive) | Cap. 5 – Séries Não-Lineares |
| 06 | Modelo STAR (Smooth Transition) | Cap. 5 – Séries Não-Lineares |
| 07 | Markov Switching (Filtro de Hamilton) | Cap. 5 – Regime Switching |
| 08 | DSGE Novo-Keynesiano (3 equações) | Cap. 6 – Modelos DSGE |
| 09 | Regularização Lasso / Ridge / Elastic Net | Cap. 7 – ML em Econometria |
| 10 | Causal Forest (Floresta Causal) | Cap. 3 e 7 |
| 11 | Modelo SAR (Spatial Autoregressive) | Cap. 8 – Econometria Espacial |
| 12 | Painel com Fatores Comuns (CCE) | Cap. 9 – Painéis N e T Grandes |
| 13 | Regressão Quantílica | Cap. 10 – Métodos Distribucionais |
| 14 | Econometria Bayesiana (MCMC / Gibbs) | Cap. 11 – Bayesiana |
| 15 | Efeitos de Pares em Redes | Cap. 12 – Econometria de Redes |
| 16 | Volatilidade Realizada e TSRV | Cap. 13 – Alta Frequência |
| 17 | Logit Misto (Mixed Logit) | Cap. 14 – Escolha Discreta |
| 18 | VAR e Vetor de Correção de Erros (VEC) | Cap. 24 – VAR e Cointegração |
| 19 | Arellano-Bond (GMM Dinâmico) | Cap. 25 – Painéis Dinâmicos |
| 20 | Regressão de Descontinuidade (Sharp RDD) | Cap. 30 – RDD |

---

## Requisitos

### Python
```bash
pip install numpy pandas scipy statsmodels scikit-learn matplotlib seaborn
pip install linearmodels arch cvxpy networkx
# Opcionais avançados:
pip install econml doubleml causalml
```

### R
```r
install.packages(c("dlm", "Synth", "DoubleML", "gmm", "tsDyn",
                   "MSGARCH", "vars", "plm", "quantreg",
                   "rjags", "spatialreg", "spdep", "mlogit",
                   "rugarch", "AER", "rdrobust", "ggplot2"))
```

---

## Como Executar

Cada pasta contém um script Python e um script R independentes. Basta executar:

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
