# DCC-GARCH – Cap. 19, pp. 181-183 | Autor: Luiz Tiago Wilcke
set.seed(42)
T <- 500
rho <- 0.3 + 0.4*sin(seq(0, 6*pi, length.out=T))
r <- matrix(0, T, 2)
h <- c(1,1)
for (t in 1:T) {
  C <- matrix(c(1, rho[t], rho[t], 1), 2, 2)
  r[t,] <- MASS::mvrnorm(1, c(0,0), diag(sqrt(h)) %*% C %*% diag(sqrt(h)))
  h <- 0.01 + 0.05*r[t,]^2 + 0.9*h
}
cat("DCC-GARCH (R) – Cap. 19, pp. 181-183\nAutor: Luiz Tiago Wilcke\n")
cat("Correlação amostral:", cor(r)[1,2], "\n")
cat("Para DCC completo use rugarch/rmgarch.\n")
