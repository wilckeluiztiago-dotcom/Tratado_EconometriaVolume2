# Modelo para o Preço do Barril de Petróleo (WTI & Brent)

**Autor:** Luiz Tiago Wilcke  

**Fonte:** *Tratado de Econometria Avançada — Volume II*  
*Teoria, Modelagem Matemática e Implementação com Python e R*

---

## Capítulo de origem (exato do livro)

**Capítulo 37 — Econometria de Commodities e Inteligência Artificial Híbrida: Previsão do Preço do Barril de Petróleo (WTI e Brent)**

Seções utilizadas:

- **37.1** Dinâmica de Preços de Commodities: Processos Contínuos  
  - 37.1.1 Modelo Heston acoplado a Ornstein-Uhlenbeck  
  - 37.1.2 Equação de Langevin em Potencial de Poço Duplo  
- **37.3** O Coração Econométrico: MS-DCC-GARCH  
  - EGARCH assimétrico e volatilidade condicional  

O modelo implementado combina o **Filtro de Kalman / Espaço de Estados** (aproximação do processo de reversão à média e nível latente do Cap. 37.1) com uma **volatilidade condicional estilo RiskMetrics/GARCH** (inspirada no Cap. 37.3).

---

## Dados reais

| Série | Código FRED | Descrição |
|-------|-------------|-----------|
| WTI   | DCOILWTICO  | Cushing, OK WTI Spot Price FOB (USD/barril) |
| Brent | DCOILBRENTEU| Europe Brent Spot Price FOB (USD/barril) |

- Frequência original: **diária**  
- Período: 20/mai/1987 → 09/set/2026 (~9.800 observações)  
- Agregação interna: **semanal** (para estabilidade numérica do Kalman)

---

## Formulação Teórica (extraída do livro)

### Processo de Ornstein-Uhlenbeck + Heston (Cap. 37.1.1)

$$
\begin{aligned}
dP_t &= \kappa(\theta - \ln P_t)\,P_t\,dt + \sqrt{v_t}\,P_t\,dW_t^S \\
dv_t &= \alpha(m - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v
\end{aligned}
$$

com \(\mathbb{E}[dW^S\,dW^v]=\rho\,dt\) (efeito de alavancagem).

### Representação em Espaço de Estados (implementada)

**Equação de Medida:**
$$
y_t = X_t + v_t, \qquad v_t\sim\mathcal{N}(0,\sigma_v^2)
$$

**Equação de Transição (Local Level):**
$$
X_t = X_{t-1} + w_t, \qquad w_t\sim\mathcal{N}(0,\sigma_w^2)
$$

onde \(X_t=\ln P_t\) é o log-preço latente “limpo”.

### Volatilidade Condicional (aprox. GARCH / RiskMetrics – Cap. 37.3)

$$
\sigma_t^2 = \lambda\sigma_{t-1}^2 + (1-\lambda)r_{t-1}^2, \qquad \lambda=0.94
$$

---

## O que o script faz

1. Baixa automaticamente WTI e Brent do FRED.  
2. Estima o modelo de Espaço de Estados por Máxima Verossimilhança (`dlm`).  
3. Aplica Filtro de Kalman + Smoother.  
4. Calcula volatilidade condicional EWMA.  
5. Gera previsão de 12 semanas com intervalo de confiança 95%.  
6. Produz 4 gráficos acadêmicos.

### Como executar

```bash
Rscript petroleo_kalman_garch.R
```

Requisitos: pacote `dlm` + acesso à internet.

---

## Arquivos gerados

| Arquivo | Descrição |
|---------|-----------|
| `petroleo_preco_kalman.png` | WTI observado × sinal suavizado (Kalman) |
| `petroleo_volatilidade.png` | Volatilidade condicional EWMA |
| `petroleo_previsao.png` | Previsão 12 semanas + IC 95% |
| `petroleo_wti_brent.png` | Comparativo WTI vs Brent (dados reais) |

---

## Referência completa

Wilcke, Luiz Tiago.  
**Tratado de Econometria Avançada — Volume II**  
Capítulo 37 (Seções 37.1 e 37.3).

---

*Modelo poderoso construído com fidelidade ao texto original do autor e dados reais de mercado.*
