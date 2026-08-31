# ============================================================
# Modelo 01 – Filtro de Kalman (Nível Local)
# Tratado de Econometria Avançada – Volume II
# Autor: Luiz Tiago Wilcke
# ============================================================

# install.packages("dlm")  # se necessário
library(dlm)

set.seed(42)
n_obs <- 120

# ------------------------------------------------------------
# 1. Dados sintéticos
# ------------------------------------------------------------
erro_transicao <- rnorm(n_obs, mean = 0, sd = 0.6)
tendencia_verdadeira <- cumsum(erro_transicao) + 10.0
ruido_medicao <- rnorm(n_obs, mean = 0, sd = 1.2)
inflacao_observada <- tendencia_verdadeira + ruido_medicao

dados <- data.frame(
  periodo = 1:n_obs,
  inflacao_observada = inflacao_observada,
  tendencia_verdadeira = tendencia_verdadeira
)

# ------------------------------------------------------------
# 2. Construção e estimação do modelo de nível local via dlm
# ------------------------------------------------------------
# Função de construção: theta = log(variâncias)
construir_nivel_local <- function(theta) {
  dlmModPoly(order = 1,
             dV = exp(theta[1]),   # H – variância de medição
             dW = exp(theta[2]))   # Q – variância de transição
}

# Valores iniciais (log)
valores_iniciais <- c(0, 0)

# Máxima verossimilhança
ajuste <- dlmMLE(inflacao_observada, parm = valores_iniciais,
                 build = construir_nivel_local)

modelo_estimado <- construir_nivel_local(ajuste$par)

cat("============================================================\n")
cat("FILTRO DE KALMAN – MODELO DE NÍVEL LOCAL (R / dlm)\n")
cat("Autor: Luiz Tiago Wilcke\n")
cat("============================================================\n")
cat("Variância de medição estimada (H):", modelo_estimado$V, "\n")
cat("Variância de transição estimada (Q):", modelo_estimado$W, "\n")

# ------------------------------------------------------------
# 3. Filtragem e suavização
# ------------------------------------------------------------
filtro <- dlmFilter(inflacao_observada, modelo_estimado)
estados_filtrados <- dropFirst(filtro$m)

suavizador <- dlmSmooth(filtro)
estados_suavizados <- dropFirst(suavizador$s)

dados$tendencia_filtrada <- estados_filtrados
dados$tendencia_suavizada <- estados_suavizados

eqm <- mean((dados$tendencia_verdadeira - dados$tendencia_filtrada)^2)
cat("Erro Quadrático Médio (EQM filtrado):", round(eqm, 4), "\n")
cat("Última estimativa filtrada:", round(tail(estados_filtrados, 1), 4), "\n")

print(head(dados[, c("periodo", "inflacao_observada",
                     "tendencia_verdadeira", "tendencia_filtrada")], 8))

# ------------------------------------------------------------
# 4. Gráfico
# ------------------------------------------------------------
png("01_Filtro_Kalman_resultado_R.png", width = 900, height = 450)
plot(dados$periodo, dados$inflacao_observada, type = "p", col = "gray60",
     pch = 19, cex = 0.6,
     main = "Filtro de Kalman – Nível Local (R)\nLuiz Tiago Wilcke",
     xlab = "Período", ylab = "Inflação / Tendência")
lines(dados$periodo, dados$tendencia_verdadeira, col = "black", lty = 2, lwd = 2)
lines(dados$periodo, dados$tendencia_filtrada, col = "blue", lwd = 2)
lines(dados$periodo, dados$tendencia_suavizada, col = "red", lwd = 1.5)
legend("topleft",
       legend = c("Observado", "Verdadeiro", "Filtrado", "Suavizado"),
       col = c("gray60", "black", "blue", "red"),
       lty = c(NA, 2, 1, 1), pch = c(19, NA, NA, NA), bty = "n")
dev.off()
cat("\nGráfico salvo: 01_Filtro_Kalman_resultado_R.png\n")
