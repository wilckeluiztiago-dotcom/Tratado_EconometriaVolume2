# Modelo 07 – Markov Switching
# Autor: Luiz Tiago Wilcke
# install.packages("MSGARCH")
set.seed(42)
n <- 250
P <- matrix(c(0.9,0.2,0.1,0.8),2,2)
mu <- c(-1,2); sig <- c(0.8,1.2)
estado <- 1; y <- numeric(n)
for (t in 1:n) {
  y[t] <- rnorm(1, mu[estado], sig[estado])
  estado <- sample(1:2, 1, prob=P[estado,])
}
cat("MARKOV SWITCHING (R)\nAutor: Luiz Tiago Wilcke\n")
cat("Série gerada com 2 regimes. Use MSGARCH::FitML para estimação completa.\n")
cat("Média da série:", mean(y), "\n")
