# Particle Filter – Cap. 41, pp. 391-397 | Autor: Luiz Tiago Wilcke
set.seed(42)
T <- 100; sig_x <- 0.3; sig_y <- 0.2
x <- numeric(T); y <- numeric(T)
for (t in 2:T) {
  x[t] <- 0.8*x[t-1] + 0.1*x[t-1]^3 + sig_x*rnorm(1)
  y[t] <- x[t] + 0.05*x[t]^2 + sig_y*rnorm(1)
}
cat("PARTICLE FILTER (R) – Cap. 41, pp. 391-397\nAutor: Luiz Tiago Wilcke\n")
cat("SIR para estado não-linear. SD estado:", sd(x), "\n")
