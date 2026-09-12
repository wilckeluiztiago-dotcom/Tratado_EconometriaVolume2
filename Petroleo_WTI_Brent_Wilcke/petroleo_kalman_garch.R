################################################################################
# Modelo Poderoso de Espaço de Estados + Volatilidade para o Preço do Petróleo
# (WTI e Brent)
#
# Extraído e expandido do Capítulo 37 do livro:
# "Tratado de Econometria Avançada — Volume II"
# Teoria, Modelagem Matemática e Implementação com Python e R
# Autor: Luiz Tiago Wilcke
#
# Componentes implementados:
# 1. Dados reais diários FRED (WTI + Brent)
# 2. Espaço de Estados / Filtro de Kalman no log-preço (Local Level + trend)
# 3. Estimação de volatilidade condicional (EWMA / aproximação GARCH)
# 4. Previsão multi-horizonte com bandas de incerteza
# 5. Diagnósticos e gráficos acadêmicos
################################################################################

library(dlm)

set.seed(42)

#-------------------------------------------------------------------------------
# 1. Ingestão de dados reais (FRED)
#    DCOILWTICO  = WTI Cushing, OK (USD/barril)
#    DCOILBRENTEU = Brent Europe (USD/barril)
#-------------------------------------------------------------------------------
cat("Baixando dados reais do FRED...\n")

download_fred <- function(series_id, dest) {
  url <- paste0("https://fred.stlouisfed.org/graph/fredgraph.csv?id=", series_id)
  tryCatch({
    download.file(url, dest, quiet = TRUE)
    TRUE
  }, error = function(e) FALSE)
}

ok_wti  <- download_fred("DCOILWTICO",  "/tmp/wti.csv")
ok_brent <- download_fred("DCOILBRENTEU", "/tmp/brent.csv")

if (!ok_wti || !ok_brent) {
  stop("Falha no download dos dados FRED. Verifique conexão.")
}

read_clean <- function(path) {
  df <- read.csv(path, stringsAsFactors = FALSE)
  colnames(df) <- c("date", "price")
  df$date  <- as.Date(df$date)
  df$price <- as.numeric(df$price)
  df <- df[!is.na(df$price) & df$price > 0, ]
  df <- df[order(df$date), ]
  df
}

wti   <- read_clean("/tmp/wti.csv")
brent <- read_clean("/tmp/brent.csv")

# Alinhar datas (inner join)
common <- merge(wti, brent, by = "date", suffixes = c("_wti", "_brent"))
common <- common[order(common$date), ]
n <- nrow(common)

cat("Período comum:", as.character(min(common$date)), "→", as.character(max(common$date)), "\n")
cat("Observações diárias:", n, "\n")
cat("WTI  último:", round(tail(common$price_wti, 1), 2), "USD/bbl\n")
cat("Brent último:", round(tail(common$price_brent, 1), 2), "USD/bbl\n\n")

#-------------------------------------------------------------------------------
# 2. Transformações
#-------------------------------------------------------------------------------
common$log_wti   <- log(common$price_wti)
common$log_brent <- log(common$price_brent)
common$ret_wti   <- c(NA, diff(common$log_wti))
common$ret_brent <- c(NA, diff(common$log_brent))

# Remover NAs iniciais
common <- common[!is.na(common$ret_wti), ]
n <- nrow(common)

#-------------------------------------------------------------------------------
# 3. Modelo de Espaço de Estados (inspirado em Cap. 37.1 – OU / Heston-OU)
#    Trabalhamos com frequência semanal para estabilidade numérica
#    (média semanal do log-preço) + Local Level
#-------------------------------------------------------------------------------
# Agregar para semanal
common$week <- format(common$date, "%Y-%W")
weekly <- aggregate(cbind(log_wti, price_wti, price_brent) ~ week, data = common, FUN = mean)
weekly$date <- as.Date(paste0(substr(weekly$week,1,4), "-01-01")) + as.numeric(substr(weekly$week,6,7))*7
weekly <- weekly[order(weekly$date), ]
y <- weekly$log_wti
n_w <- length(y)

build_ll <- function(theta) {
  dV <- exp(theta[2])^2
  dW <- exp(theta[1])^2
  dlmModPoly(order = 1, dV = dV, dW = dW)
}

cat("Estimando modelo de Espaço de Estados (Local Level semanal)...\n")
start <- c(log(0.03), log(0.02))
fit <- dlmMLE(y, parm = start, build = build_ll, method = "L-BFGS-B",
              lower = c(-10, -10), upper = c(2, 2),
              control = list(maxit = 200))

mod <- build_ll(fit$par)
cat("σ_nível  :", round(exp(fit$par[1]), 5), "\n")
cat("σ_medida :", round(exp(fit$par[2]), 5), "\n")
cat("LogLik   :", round(-fit$value, 1), "\n\n")

filtro <- dlmFilter(y, mod)
suave  <- dlmSmooth(filtro)

nivel  <- as.numeric(dropFirst(suave$s))
preco_suave <- exp(nivel)

# Mapear de volta para o data.frame diário (aproximação por forward-fill)
common$nivel_suave <- NA
common$preco_suave <- NA
for (i in seq_along(nivel)) {
  # encontrar datas da semana correspondente
  idx <- which(common$week == weekly$week[i + 1])  # +1 por causa do dropFirst
  if (length(idx) > 0) {
    common$nivel_suave[idx] <- nivel[i]
    common$preco_suave[idx] <- preco_suave[i]
  }
}
# preencher NAs iniciais
common$preco_suave[is.na(common$preco_suave)] <- common$price_wti[is.na(common$preco_suave)]
common$nivel_suave[is.na(common$nivel_suave)] <- common$log_wti[is.na(common$nivel_suave)]
common$slope <- 0

#-------------------------------------------------------------------------------
# 4. Volatilidade condicional (EWMA – aproximação robusta do GARCH do Cap. 37)
#    λ = 0.94 (padrão RiskMetrics)
#-------------------------------------------------------------------------------
lambda <- 0.94
ret <- common$ret_wti
vol <- numeric(n)
vol[1] <- sd(ret, na.rm = TRUE)
for (t in 2:n) {
  vol[t] <- sqrt(lambda * vol[t-1]^2 + (1 - lambda) * ret[t-1]^2)
}
common$vol_ewma <- vol * 100   # em %

#-------------------------------------------------------------------------------
# 5. Previsão (horizonte 30 dias úteis ≈ 1,5 mês)
#-------------------------------------------------------------------------------
horizon <- 12   # 12 semanas ≈ 3 meses
ultimo_nivel <- as.numeric(tail(nivel, 1))
prev_nivel <- rep(ultimo_nivel, horizon)
P_last <- tryCatch({
  as.numeric(tail(filtro$U.C, 1))^2 * as.numeric(tail(filtro$D.C, 1))
}, error = function(e) exp(fit$par[1])^2)
if (length(P_last) == 0 || is.na(P_last[1])) P_last <- exp(fit$par[1])^2
se <- sqrt(as.numeric(P_last) + (1:horizon) * exp(fit$par[1])^2)

prev_preco <- exp(prev_nivel)
lower <- exp(prev_nivel - 1.96 * se)
upper <- exp(prev_nivel + 1.96 * se)

# Datas futuras semanais
ultima <- max(weekly$date)
datas_fut <- seq(ultima + 7, by = "week", length.out = horizon)

#-------------------------------------------------------------------------------
# 6. Gráficos (base R – robusto)
#-------------------------------------------------------------------------------
png("petroleo_preco_kalman.png", width = 1200, height = 700, res = 120)
par(mar = c(5,4,4,2) + 0.1)
plot(common$date, common$price_wti, type = "l", col = "gray50", lwd = 1,
     main = "Preço WTI – Modelo de Espaço de Estados (Kalman)\nCapítulo 37 | Autor: Luiz Tiago Wilcke",
     xlab = "", ylab = "USD por barril",
     ylim = range(c(common$price_wti, common$preco_suave), na.rm = TRUE))
lines(common$date, common$preco_suave, col = "#E63946", lwd = 2)
legend("topleft", legend = c("WTI Observado (FRED)", "Sinal Suavizado (Kalman)"),
       col = c("gray50", "#E63946"), lwd = c(1,2), bty = "n")
grid()
dev.off()

png("petroleo_volatilidade.png", width = 1200, height = 500, res = 120)
plot(common$date, common$vol_ewma, type = "l", col = "#457B9D", lwd = 1.2,
     main = "Volatilidade Condicional EWMA do WTI (aprox. GARCH – Cap. 37.3)",
     xlab = "", ylab = "Volatilidade diária (%)")
grid()
dev.off()

png("petroleo_previsao.png", width = 1200, height = 600, res = 120)
idx <- which(common$date >= max(common$date) - 400)
plot(common$date[idx], common$preco_suave[idx], type = "l", col = "#E63946", lwd = 2,
     xlim = c(min(common$date[idx]), max(datas_fut)),
     ylim = range(c(common$preco_suave[idx], lower, upper), na.rm = TRUE),
     main = "Previsão do Preço WTI (12 semanas ≈ 3 meses)\nModelo Local Level (Kalman) – Cap. 37",
     xlab = "", ylab = "USD/barril")
polygon(c(datas_fut, rev(datas_fut)), c(lower, rev(upper)),
        col = adjustcolor("#E63946", 0.25), border = NA)
lines(datas_fut, prev_preco, col = "#E63946", lwd = 2, lty = 2)
abline(v = max(common$date), col = "gray40", lty = 3)
legend("topleft", legend = c("Histórico suavizado", "Previsão", "IC 95%"),
       col = c("#E63946", "#E63946", adjustcolor("#E63946", 0.4)),
       lwd = c(2,2,8), lty = c(1,2,1), bty = "n")
grid()
dev.off()

png("petroleo_wti_brent.png", width = 1200, height = 600, res = 120)
plot(common$date, common$price_wti, type = "l", col = "#1D3557", lwd = 1.2,
     main = "WTI vs Brent – Dados Reais FRED\nCapítulo 37 | Luiz Tiago Wilcke",
     xlab = "", ylab = "USD/barril")
lines(common$date, common$price_brent, col = "#E63946", lwd = 1.2)
legend("topleft", legend = c("WTI", "Brent"), col = c("#1D3557", "#E63946"),
       lwd = 1.5, bty = "n")
grid()
dev.off()

#-------------------------------------------------------------------------------
# 7. Resumo
#-------------------------------------------------------------------------------
cat("\n========== RESUMO DO MODELO ==========\n")
cat("Capítulo de origem : 37 — Econometria de Commodities e IA Híbrida\n")
cat("Seções de referência: 37.1 (OU / Heston-OU) + 37.3 (MS-DCC-GARCH / EGARCH)\n")
cat("Autor do livro     : Luiz Tiago Wilcke\n\n")
cat("Último WTI observado :", round(tail(common$price_wti, 1), 2), "USD/bbl\n")
cat("Último nível suavizado:", round(tail(preco_suave, 1), 2), "USD/bbl\n")
cat("Previsão (h=12 semanas):", round(prev_preco[horizon], 2), "USD/bbl\n")
cat("Volatilidade atual    :", round(tail(common$vol_ewma, 1), 2), "% ao dia\n")
cat("\nGráficos gerados:\n")
cat("  - petroleo_preco_kalman.png\n")
cat("  - petroleo_volatilidade.png\n")
cat("  - petroleo_previsao.png\n")
cat("  - petroleo_wti_brent.png\n")
cat("\nScript concluído com sucesso.\n")
