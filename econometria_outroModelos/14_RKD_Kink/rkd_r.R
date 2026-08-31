# RKD – Cap. 34, pp. 326-333 | Autor: Luiz Tiago Wilcke
set.seed(42)
n <- 2000; x <- runif(n, -1, 1); c <- 0
b <- ifelse(x < c, 0.5*x, 0.5*x + 1.0*(x-c))
Y <- 2*b + 1 + 0.3*x + rnorm(n,0,0.5)
cat("RKD (R) – Cap. 34, pp. 326-333\nAutor: Luiz Tiago Wilcke\n")
cat("Identifica efeito via mudança de slope no limiar.\n")
