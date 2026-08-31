# ============================================================
# Modelo 02 – Controle Sintético (Synthetic Control)
# Tratado de Econometria Avançada – Volume II
# Autor: Luiz Tiago Wilcke
# ============================================================

# install.packages("Synth")
library(Synth)

set.seed(123)

# ------------------------------------------------------------
# 1. Dados sintéticos
# ------------------------------------------------------------
n_unidades <- 11
n_periodos <- 30
periodo_tratamento <- 20

fator_comum <- cumsum(rnorm(n_periodos, 0, 0.3))
resultado_mat <- matrix(0, n_unidades, n_periodos)

for (i in 1:n_unidades) {
  nivel <- 50 + (i-1)*2 + rnorm(1, 0, 1)
  resultado_mat[i, ] <- nivel + fator_comum + rnorm(n_periodos, 0, 1.5)
}
# Efeito do tratamento
efeito_verdadeiro <- 8
resultado_mat[1, (periodo_tratamento+1):n_periodos] <- 
  resultado_mat[1, (periodo_tratamento+1):n_periodos] + efeito_verdadeiro

# Formato longo para Synth
dados_longo <- data.frame(
  unidade = rep(0:(n_unidades-1), each = n_periodos),
  periodo = rep(1:n_periodos, times = n_unidades),
  resultado = as.vector(t(resultado_mat))
)
# Adicionar preditores pré-tratamento (médias)
dados_longo$preditor1 <- ave(dados_longo$resultado, dados_longo$unidade, FUN = function(x) mean(x[1:periodo_tratamento]))

# ------------------------------------------------------------
# 2. Preparação e estimação com Synth
# ------------------------------------------------------------
dataprep_out <- dataprep(
  foo = dados_longo,
  predictors = c("preditor1"),
  predictors.op = "mean",
  dependent = "resultado",
  unit.variable = "unidade",
  time.variable = "periodo",
  treatment.identifier = 0,
  controls.identifier = 1:(n_unidades-1),
  time.predictors.prior = 1:periodo_tratamento,
  time.optimize.ssr = 1:periodo_tratamento,
  time.plot = 1:n_periodos
)

synth_out <- synth(dataprep_out)

cat("============================================================\n")
cat("CONTROLE SINTÉTICO (R / Synth)\n")
cat("Autor: Luiz Tiago Wilcke\n")
cat("============================================================\n")
cat("Pesos das unidades doadoras:\n")
print(round(synth_out$solution.w, 4))

# Efeito
y_tratada <- dataprep_out$Y1plot
y_sintetico <- dataprep_out$Y0plot %*% synth_out$solution.w
efeito <- y_tratada - y_sintetico
cat("\nEfeito médio pós-tratamento:", round(mean(efeito[(periodo_tratamento+1):n_periodos]), 3), "\n")
cat("Efeito verdadeiro (simulado):", efeito_verdadeiro, "\n")

# Gráfico
png("02_Controle_Sintetico_resultado_R.png", width = 900, height = 450)
path.plot(synth.res = synth_out, dataprep.res = dataprep_out,
          Ylab = "Resultado", Xlab = "Período",
          Legend = c("Tratada", "Sintético"),
          Main = "Controle Sintético – Luiz Tiago Wilcke")
abline(v = periodo_tratamento, lty = 2, col = "gray")
dev.off()
cat("\nGráfico salvo: 02_Controle_Sintetico_resultado_R.png\n")
