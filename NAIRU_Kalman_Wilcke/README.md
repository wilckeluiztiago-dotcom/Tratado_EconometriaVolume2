# Modelo de Espaço de Estados para Estimação e Previsão da NAIRU

**Autor:** Luiz Tiago Wilcke  

**Fonte:** *Tratado de Econometria Avançada — Volume II*  
*Teoria, Modelagem Matemática e Implementação com Python e R*

---

## Capítulo de origem

**Capítulo 36 — Modelagem Econométrica Multivariada e Redução de Dimensionalidade Aplicada à Dinâmica do Desemprego**

Seção específica: **36.3 Estimação de Estados Não-Observáveis: O NAIRU via Filtro de Kalman**

Este repositório implementa em **R** o modelo de espaço de estados apresentado no livro para estimar a **Taxa de Desemprego Não-Aceleradora da Inflação (NAIRU)** — uma variável latente fundamental na macroeconomia — e gerar previsões do nível estrutural de desemprego de um país.

---

## Formulação Teórica do Modelo

O modelo é composto por duas equações centrais (Gordon’s triangular Phillips curve + random-walk transition):

### Equação de Medida (Curva de Phillips)

$$
\Delta \pi_t = \beta_1 \Delta \pi_{t-1} + \beta_2 \Delta \pi_{t-2} + \gamma (u_t - u_t^*) + \delta z_t + \epsilon_t, \qquad \epsilon_t \sim \mathcal{N}(0, \sigma_\epsilon^2)
$$

onde:

- \( u_t \) = taxa de desemprego observada  
- \( u_t^* \) = NAIRU (estado não-observável)  
- \( z_t \) = vetor de choques de oferta  
- \( \epsilon_t \) = choque de demanda idiossincrático

### Equação de Transição (Dinâmica da NAIRU)

$$
u_t^* = u_{t-1}^* + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, \sigma_\eta^2)
$$

Os erros de medida e de transição são ortogonais:

$$
\mathbb{E}[\epsilon_t \eta_s] = 0 \quad \forall\, t,s
$$

### Algoritmo do Filtro de Kalman

**1. Predição (a priori)**

$$
x_{t|t-1} = x_{t-1|t-1}
$$

$$
P_{t|t-1} = P_{t-1|t-1} + \sigma_\eta^2
$$

**2. Atualização (a posteriori)**

Ganho de Kalman:

$$
K_t = \frac{\gamma P_{t|t-1}}{\gamma^2 P_{t|t-1} + \sigma_\epsilon^2}
$$

Atualização do estado:

$$
x_{t|t} = x_{t|t-1} + K_t \Bigl[ \Delta\pi_t - \bigl(\beta_1\Delta\pi_{t-1} + \beta_2\Delta\pi_{t-2} + \gamma(u_t - x_{t|t-1}) + \delta z_t\bigr) \Bigr]
$$

---

## Implementação em R

O script `nairu_kalman_desemprego.R` realiza:

1. Simulação de dados macroeconômicos realistas (desemprego + variação da inflação) com base nos parâmetros do livro.
2. Estimação por máxima verossimilhança das variâncias \(\sigma_\epsilon^2\) e \(\sigma_\eta^2\) via pacote `dlm`.
3. Aplicação do **Filtro de Kalman** e do **Smoother de Kalman**.
4. Geração de três gráficos acadêmicos:
   - Trajetória da NAIRU estimada vs. desemprego observado
   - Hiato do desemprego \(u_t - u_t^*\)
   - Previsão da NAIRU para os próximos 12 meses com intervalo de confiança

### Como executar

```bash
Rscript nairu_kalman_desemprego.R
```

Requisitos: pacotes `dlm` e `ggplot2`.

---

## Arquivos gerados

| Arquivo | Descrição |
|---------|-----------|
| `nairu_desemprego_kalman.png` / `.pdf` | Gráfico principal: desemprego observado × NAIRU filtrada/suavizada |
| `hiato_desemprego.png` | Hiato do desemprego (pressão inflacionária/deflacionária) |
| `previsao_nairu.png` | Previsão da NAIRU (horizonte de 12 meses) |

---

## Referência completa

Wilcke, Luiz Tiago.  
**Tratado de Econometria Avançada — Volume II**  
*Teoria, Modelagem Matemática e Implementação com Python e R*  
Capítulo 36, Seção 36.3.

---

*Modelo extraído e adaptado com fidelidade ao texto original do autor.*
