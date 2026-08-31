# Modelo 08 – DSGE
# Autor: Luiz Tiago Wilcke
set.seed(42)
T <- 100
y_gap <- pi <- i <- numeric(T)
shock <- rnorm(T, 0, 0.5)
kappa <- 0.1; phi_pi <- 1.5; phi_y <- 0.5
for (t in 2:T) {
  y_gap[t] <- 0.7*y_gap[t-1] - 0.3*(i[t-1]-pi[t-1]) + shock[t]
  pi[t] <- 0.6*pi[t-1] + kappa*y_gap[t] + 0.1*rnorm(1)
  i[t] <- phi_pi*pi[t] + phi_y*y_gap[t] + 0.2*rnorm(1)
}
cat("DSGE NOVO-KEYNESIANO (R)\nAutor: Luiz Tiago Wilcke\n")
cat("Médias: hiato=", mean(y_gap), " inflação=", mean(pi), " juros=", mean(i), "\n")
