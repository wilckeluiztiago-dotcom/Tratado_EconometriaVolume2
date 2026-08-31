# ============================================================
# Modelo 05 – TAR
# Autor: Luiz Tiago Wilcke
# ============================================================

# install.packages("tsDyn")
library(tsDyn)

set.seed(42)
n <- 300
limiar_verdadeiro <- 0.5
y <- numeric(n)
y[1] <- 0
for (t in 2:n) {
  if (y[t-1] <= limiar_verdadeiro) {
    y[t] <- 0.3 + 0.6 * y[t-1] + rnorm(1, 0, 0.8)
  } else {
    y[t] <- -0.2 + 0.3 * y[t-1] + rnorm(1, 0, 0.8)
  }
}

modelo <- setar(y, m = 1, thDelay = 1)
cat("============================================================\n")
cat("MODELO TAR (R / tsDyn)\nAutor: Luiz Tiago Wilcke\n")
cat("============================================================\n")
print(summary(modelo))
cat("\nLimiar verdadeiro:", limiar_verdadeiro, "\n")
