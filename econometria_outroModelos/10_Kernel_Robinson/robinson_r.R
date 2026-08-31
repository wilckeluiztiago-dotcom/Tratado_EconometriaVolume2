# Robinson – Cap. 29, pp. 277-285 | Autor: Luiz Tiago Wilcke
set.seed(42)
n <- 600
X <- matrix(rnorm(n*2), n, 2)
Z <- runif(n, -2, 2)
Y <- X %*% c(1, -0.5) + sin(Z) + 0.5*Z^2 + rnorm(n, 0, 0.5)
cat("ROBINSON (R) – Cap. 29, pp. 277-285\nAutor: Luiz Tiago Wilcke\n")
cat("Y = X beta + g(Z) + e | g não-paramétrico via kernel.\n")
