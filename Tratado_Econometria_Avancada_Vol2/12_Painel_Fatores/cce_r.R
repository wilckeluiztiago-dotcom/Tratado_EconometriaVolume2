# Modelo 12 – CCE
# Autor: Luiz Tiago Wilcke
set.seed(42)
N <- 30; T <- 40
f <- matrix(rnorm(T*2), T, 2)
lambda <- matrix(rnorm(N*2), N, 2)
beta_v <- 1.5
x <- matrix(rnorm(N*T), N, T)
y <- matrix(0, N, T)
for (i in 1:N) y[i,] <- beta_v*x[i,] + lambda[i,] %*% t(f) + rnorm(T)
y_bar <- colMeans(y); x_bar <- colMeans(x)
betas <- sapply(1:N, function(i) {
  Z <- cbind(x[i,], y_bar, x_bar)
  coef(lm(y[i,] ~ Z - 1))[1]
})
cat("CCE (R)\nAutor: Luiz Tiago Wilcke\n")
cat("Beta CCE médio:", mean(betas), " verdadeiro: 1.5\n")
