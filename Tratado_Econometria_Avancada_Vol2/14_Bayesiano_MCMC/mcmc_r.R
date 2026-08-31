# Modelo 14 – MCMC
# Autor: Luiz Tiago Wilcke
set.seed(42)
n <- 200
X <- cbind(1, rnorm(n))
beta_v <- c(1,2)
y <- X %*% beta_v + rnorm(n)
# Gibbs simples
n_iter <- 3000; burn <- 1000
beta_chain <- matrix(0, n_iter, 2)
sig2 <- 1; beta <- c(0,0)
XtX <- t(X)%*%X
for (i in 1:n_iter) {
  V <- solve(XtX/sig2 + diag(2)/100)
  m <- V %*% (t(X)%*%y / sig2)
  beta <- mvrnorm <- as.vector(m + t(chol(V)) %*% rnorm(2))
  resid <- y - X%*%beta
  a <- 0.01 + n/2; b <- 0.01 + 0.5*sum(resid^2)
  sig2 <- 1/rgamma(1, a, rate=b)
  beta_chain[i,] <- beta
}
post <- beta_chain[(burn+1):n_iter,]
cat("MCMC BAYESIANO (R)\nAutor: Luiz Tiago Wilcke\n")
cat("Média a posteriori:", colMeans(post), "\n")
cat("Verdadeiro: 1 2\n")
