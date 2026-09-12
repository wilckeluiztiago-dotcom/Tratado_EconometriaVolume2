# Modelo de Espaço de Estados para o Preço Global do Trigo

**Autor:** Luiz Tiago Wilcke  

**Fonte:** *Tratado de Econometria Avançada — Volume II*  
*Teoria, Modelagem Matemática e Implementação com Python e R*

---

## Capítulo de origem

**Capítulo 37 — Econometria de Commodities e Inteligência Artificial Híbrida: Previsão do Preço do Barril de Petróleo (WTI e Brent)**

Seção de referência principal: **37.1 Dinâmica de Preços de Commodities: Processos Contínuos**  
(Modelo de Heston acoplado a Ornstein-Uhlenbeck e Equação de Langevin)

O modelo foi adaptado para o **preço global do trigo** (commodity agrícola) mantendo a mesma estrutura teórica de processos de difusão com reversão à média.

---

## Dados reais utilizados

- **Série:** Global price of Wheat (PWHEAMTUSDM)  
- **Fonte:** Federal Reserve Economic Data (FRED)  
- **Unidade:** Dólares americanos por tonelada métrica  
- **Frequência:** Mensal  
- **Período:** Janeiro/1992 – Julho/2026 (415 observações)

---

## Formulação Teórica do Modelo

### Processo de Ornstein-Uhlenbeck (reversão à média) — Cap. 37.1.1

No tempo contínuo, o log-preço \(X_t = \ln P_t\) segue:

$$
dX_t = \kappa(\theta - X_t)\,dt + \sigma\,dW_t
$$

onde:

- \(\kappa > 0\) = velocidade de reversão à média  
- \(\theta\) = nível de equilíbrio de longo prazo (custo marginal / equilíbrio de mercado)  
- \(\sigma\) = volatilidade instantânea  

### Discretização (Euler / AR(1))

$$
X_t = (1-\phi)\theta + \phi X_{t-1} + \varepsilon_t, \qquad \phi = e^{-\kappa\Delta t}
$$

### Representação em Espaço de Estados (Filtro de Kalman)

**Equação de Medida:**

$$
y_t = X_t + v_t, \qquad v_t \sim \mathcal{N}(0, \sigma_v^2)
$$

**Equação de Transição (aproximação Local Level + persistência):**

$$
X_t = X_{t-1} + w_t, \qquad w_t \sim \mathcal{N}(0, \sigma_w^2)
$$

(O Local Level captura a persistência elevada típica de preços de commodities e é numericamente estável; o parâmetro de reversão é recuperado via diagnóstico de autocorrelação residual.)

### Modelo de Heston-OU completo (referência do livro)

$$
\begin{aligned}
dP_t &= \kappa(\theta - \ln P_t)P_t\,dt + \sqrt{v_t}\,P_t\,dW_t^S \\
dv_t &= \alpha(m - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v
\end{aligned}
$$

com \(\mathbb{E}[dW^S dW^v] = \rho\,dt\) (efeito de alavancagem).

---

## Implementação em R

O script `preco_trigo_kalman.R`:

1. Baixa dados reais do FRED automaticamente.
2. Estima o modelo de Espaço de Estados por Máxima Verossimilhança (`dlm`).
3. Aplica o **Filtro de Kalman** e o **Smoother**.
4. Gera previsão de 12 meses com intervalo de confiança.
5. Produz três gráficos acadêmicos.

### Como executar

```bash
Rscript preco_trigo_kalman.R
```

Requisitos: `dlm`, `ggplot2` (e acesso à internet para o download do FRED).

---

## Arquivos gerados

| Arquivo | Descrição |
|---------|-----------|
| `trigo_preco_kalman.png` / `.pdf` | Preço observado × sinal suavizado pelo Kalman |
| `trigo_log_nivel.png` | Log-preço e nível latente estimado |
| `trigo_previsao.png` | Previsão 12 meses + intervalo 95% |

---

## Referência completa

Wilcke, Luiz Tiago.  
**Tratado de Econometria Avançada — Volume II**  
*Teoria, Modelagem Matemática e Implementação com Python e R*  
Capítulo 37, Seção 37.1.

---

*Modelo extraído e adaptado com fidelidade ao texto original do autor, aplicado a dados reais de commodity agrícola.*
