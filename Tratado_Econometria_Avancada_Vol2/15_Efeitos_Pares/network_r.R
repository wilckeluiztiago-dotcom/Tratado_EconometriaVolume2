# Modelo 15 – Efeitos de Pares
# Autor: Luiz Tiago Wilcke
set.seed(42)
n <- 100
A <- matrix(runif(n*n)<0.1, n, n); diag(A) <- 0; A <- (A+t(A))/2
grau <- rowSums(A); grau[grau==0] <- 1
G <- A / grau
beta_v <- 0.4; gamma_v <- 1; delta_v <- 0.3
x <- rnorm(n); e <- rnorm(n)
y <- solve(diag(n)-beta_v*G) %*% (gamma_v*x + delta_v*G%*%x + e)
Gx <- G%*%x; G2x <- G%*%Gx; Gy <- G%*%y
Z <- cbind(1, x, Gx, G2x)
Gy_hat <- Z %*% solve(t(Z)%*%Z) %*% t(Z) %*% Gy
X2 <- cbind(1, Gy_hat, x, Gx)
theta <- solve(t(X2)%*%X2) %*% t(X2) %*% y
cat("EFEITOS DE PARES (R)\nAutor: Luiz Tiago Wilcke\n")
cat("beta estimado:", theta[2], " verdadeiro: 0.4\n")
