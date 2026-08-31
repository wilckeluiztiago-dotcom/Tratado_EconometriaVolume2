# Econometria – Outros Modelos Avançados

**Tratado de Econometria Avançada – Volume II**  
**Autor:** Luiz Tiago Wilcke

---

## Descrição

Pacote com **20 modelos econométricos avançados e completos**, com implementações mais complexas em Python e R. Cada modelo indica o **capítulo e as páginas** do livro de origem.

---

## Lista dos 20 Modelos e Páginas de Referência

| Nº | Modelo | Capítulo | Páginas |
|----|--------|----------|---------|
| 01 | BLP – Demanda estrutural (inversão de contração + GMM) | Cap. 14 | pp. 127–138 (inversão: 129–130) |
| 02 | NFXP – Nested Fixed Point de Rust (1987) | Cap. 15 | pp. 139–150 (NFXP: 140–141) |
| 03 | CCP – Hotz e Miller (1993) | Cap. 15 | pp. 141–142 |
| 04 | DCC-GARCH (Engle 2002) | Cap. 19 | pp. 181–183 (DCC: 182) |
| 05 | Cópulas e dependência de caudas (Clayton / Sklar) | Cap. 19 | pp. 183–184 |
| 06 | LSTM – Long Short-Term Memory (do zero) | Cap. 20 | pp. 192–197 (célula: 193–194) |
| 07 | SEM e mediação causal (bootstrap) | Cap. 21 | pp. 202–211 (mediação: 204–205) |
| 08 | Tobit e seleção de Heckman (Heckit) | Cap. 27 | pp. 257–265 |
| 09 | Poisson e ZIP (Zero-Inflated Poisson) | Cap. 28 | pp. 267–275 (ZIP: 269–270) |
| 10 | Robinson (1988) – regressão parcialmente linear | Cap. 29 | pp. 277–285 (Robinson: 279–280) |
| 11 | DiD – TWFE vs Callaway-Sant'Anna (2021) | Cap. 31 | pp. 297–305 (CS: 300–301) |
| 12 | Matching, IPW e estimador duplamente robusto | Cap. 32 | pp. 306–314 (DR: 308) |
| 13 | Matrix Completion / GSC (norma nuclear) | Cap. 33 | pp. 315–324 (317–318) |
| 14 | Regression Kink Design (RKD) | Cap. 34 | pp. 326–333 |
| 15 | Synthetic Difference-in-Differences (SDID) | Cap. 35 | pp. 334–342 |
| 16 | Cox – riscos proporcionais | Cap. 26 | pp. 248–256 (Cox: 250–251) |
| 17 | Equações simultâneas – 2SLS | Cap. 23 | pp. 222–230 (2SLS: 224) |
| 18 | GPV – leilões estruturais (Guerre-Perrigne-Vuong) | Cap. 18 | pp. 173–180 (GPV: 174–175) |
| 19 | Econometria de texto – EPU + LDA | Cap. 17 | pp. 163–172 (EPU: 164–165; LDA: 165) |
| 20 | Filtro de partículas (SIR) para DSGE não-linear | Cap. 41 | pp. 391–397 (392–393) |

---

## Estrutura

```
econometria_outroModelos/
├── README.md
├── 01_BLP_Demanda/
├── 02_NFXP_Rust/
├── 03_CCP_HotzMiller/
├── 04_DCC_GARCH/
├── 05_Copulas_Dinamicas/
├── 06_LSTM_Series/
├── 07_SEM_Mediacao/
├── 08_Tobit_Heckman/
├── 09_Poisson_ZIP/
├── 10_Kernel_Robinson/
├── 11_DiD_Callaway/
├── 12_Matching_IPW_DR/
├── 13_GSC_MatrixCompletion/
├── 14_RKD_Kink/
├── 15_SDID/
├── 16_Cox_Sobrevivencia/
├── 17_Equacoes_Simultaneas/
├── 18_Auction_GPV/
├── 19_Text_EPU_LDA/
└── 20_Particle_Filter_DSGE/
```

Cada pasta contém scripts Python e R com:
- Geração de dados sintéticos coerentes com o DGP do modelo
- Estimação completa (não apenas esboço)
- Saída interpretável com comparação ao valor verdadeiro
- Citação explícita de capítulo e páginas no cabeçalho

---

## Destaques de complexidade

- **BLP:** inversão de ponto fixo por mercado + GMM de segundo estágio com instrumentos de rivalidade
- **NFXP:** nested fixed point completo (EV + MLE) no modelo de Rust
- **DCC-GARCH:** duas etapas (GARCH univariado + DCC)
- **LSTM:** célula implementada do zero (portas f, i, o, g)
- **Heckman:** probit + inverso de Mills + MQO
- **Callaway-Sant'Anna:** ATT(g,t) com controles limpos sob adoção escalonada
- **Matrix Completion:** Soft-Impute com soft-thresholding de SVD
- **GPV:** recuperação não-paramétrica de valores a partir de lances
- **LDA:** Gibbs sampling para tópicos
- **Particle Filter:** SIR com reamostragem sistemática

---

## Como executar

```bash
python 01_BLP_Demanda/blp_python.py
Rscript 16_Cox_Sobrevivencia/cox_r.R
```

---

## Citação

Wilcke, Luiz Tiago. *Tratado de Econometria Avançada – Volume II*. 2024.

**Autor:** Luiz Tiago Wilcke  
**Licença:** uso didático e de pesquisa com atribuição.
