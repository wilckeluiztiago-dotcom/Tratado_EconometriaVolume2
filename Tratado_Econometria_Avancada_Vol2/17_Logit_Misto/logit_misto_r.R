# Modelo 17 – Logit Misto
# Autor: Luiz Tiago Wilcke
set.seed(42)
n <- 200; J <- 3
x <- matrix(rnorm(n*J), n, J)
b_v <- 1; sigma_v <- 0.8
beta_i <- rnorm(n, b_v, sigma_v)
util <- beta_i * x
prob <- exp(util) / rowSums(exp(util))
escolha <- sapply(1:n, function(i) sample(1:J, 1, prob=prob[i,]))
cat("LOGIT MISTO (R)\nAutor: Luiz Tiago Wilcke\n")
cat("Dados gerados. Use mlogit ou gmnl para estimação completa.\n")
cat("Proporção de escolhas por alternativa:", table(escolha)/n, "\n")
