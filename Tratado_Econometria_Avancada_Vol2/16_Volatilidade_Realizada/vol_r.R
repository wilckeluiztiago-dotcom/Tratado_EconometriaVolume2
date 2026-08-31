# Modelo 16 – Volatilidade Realizada
# Autor: Luiz Tiago Wilcke
set.seed(42)
n_dias <- 20; n_ticks <- 390
rv <- tsrv <- numeric(n_dias)
for (d in 1:n_dias) {
  dW <- rnorm(n_ticks, 0, 0.02/sqrt(n_ticks))
  log_p <- cumsum(dW) + rnorm(n_ticks, 0, 0.001)
  ret <- diff(log_p)
  rv[d] <- sum(ret^2)
  ret_g <- diff(log_p[seq(1,n_ticks,5)])
  tsrv[d] <- sum(ret_g^2) - (n_ticks/5)/n_ticks * sum(ret^2)
}
cat("VOLATILIDADE REALIZADA (R)\nAutor: Luiz Tiago Wilcke\n")
cat("RV média:", mean(rv), " TSRV média:", mean(tsrv), "\n")
