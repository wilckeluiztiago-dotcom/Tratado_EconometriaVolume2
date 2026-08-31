# ============================================================
# Modelo 04 – GMM
# Autor: Luiz Tiago Wilcke
# ============================================================

# install.packages("gmm")
library(gmm)

set.seed(42)
n <- 500
x_latente <- rnorm(n)
e <- 0.7 * x_latente + rnorm(n)
x <- x_latente + rnorm(n, 0, 0.5)
z1 <- 0.8 * x_latente + rnorm(n)
z2 <- 0.5 * x_latente + rnorm(n)
y <- 1 + 2 * x + e

dados <- data.frame(resultado = y, explicativa = x, instrumento1 = z1, instrumento2 = z2)

# Momentos: E[e] = 0, E[e*z1] = 0, E[e*z2] = 0
g <- function(theta, dados) {
  e <- dados$resultado - theta[1] - theta[2] * dados$explicativa
  cbind(e, e * dados$instrumento1, e * dados$instrumento2)
}

ajuste <- gmm(g, x = dados, t0 = c(0, 0))
cat("============================================================\n")
cat("GMM – R (pacote gmm)\nAutor: Luiz Tiago Wilcke\n")
cat("============================================================\n")
print(summary(ajuste))
