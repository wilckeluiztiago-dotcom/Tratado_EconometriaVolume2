# Modelo 20 – Sharp RDD
# Autor: Luiz Tiago Wilcke
# install.packages("rdrobust")
set.seed(42)
n <- 1000
x <- runif(n, -1, 1)
D <- as.numeric(x >= 0)
y <- 1 + 0.5*x + 2*D + rnorm(n)
h <- 0.3
esq <- x >= -h & x < 0
dir <- x >= 0 & x <= h
be <- coef(lm(y[esq] ~ x[esq]))
bd <- coef(lm(y[dir] ~ x[dir]))
efeito <- (bd[1] + bd[2]*0) - (be[1] + be[2]*0)
cat("SHARP RDD (R)\nAutor: Luiz Tiago Wilcke\n")
cat("Efeito estimado:", efeito, " verdadeiro: 2.0\n")
