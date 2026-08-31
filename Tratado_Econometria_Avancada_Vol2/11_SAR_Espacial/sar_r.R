# Modelo 11 – SAR
# Autor: Luiz Tiago Wilcke
set.seed(42)
n <- 50
W <- matrix(runif(n*n), n, n); W <- (W+t(W))/2; diag(W) <- 0
W <- W / rowSums(W)
rho <- 0.5; beta <- c(1,2)
X <- cbind(1, rnorm(n))
y <- solve(diag(n)-rho*W) %*% (X%*%beta + rnorm(n))
# 2SLS
Wy <- W%*%y
Z <- cbind(X, W%*%X[,2])
Wy_hat <- Z %*% solve(t(Z)%*%Z) %*% t(Z) %*% Wy
X2 <- cbind(X, Wy_hat)
theta <- solve(t(X2)%*%X2) %*% t(X2) %*% y
cat("MODELO SAR (R)\nAutor: Luiz Tiago Wilcke\n")
cat("rho estimado:", theta[3], " verdadeiro: 0.5\n")
cat("beta:", theta[1:2], "\n")
