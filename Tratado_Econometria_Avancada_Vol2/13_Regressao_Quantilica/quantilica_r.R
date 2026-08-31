# Modelo 13 – Quantílica
# Autor: Luiz Tiago Wilcke
library(quantreg)
set.seed(42)
n <- 500
x <- runif(n, 0, 10)
y <- 2 + 1*x + (0.5+0.3*x)*rnorm(n)
cat("REGRESSÃO QUANTÍLICA (R)\nAutor: Luiz Tiago Wilcke\n")
for (tau in c(0.25,0.5,0.75,0.9)) {
  m <- rq(y ~ x, tau=tau)
  cat("Tau=", tau, " coef=", coef(m), "\n")
}
