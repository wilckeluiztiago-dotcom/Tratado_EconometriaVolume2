# BLP – Cap. 14, pp. 127-138 | Autor: Luiz Tiago Wilcke
set.seed(42)
J <- 5; Tm <- 20; n_cons <- 100
X <- array(runif(Tm*J*3), dim=c(Tm,J,3))
X[,,1] <- 1
preco <- 0.5 + 0.3*X[,,3] + rnorm(Tm*J,0,0.1) + 0.4
X[,,2] <- preco
b <- c(-1.5, -2.0, 1.0)
xi <- matrix(rnorm(Tm*J,0,0.3), Tm, J)
# Shares via mixed logit simplificado
shares <- matrix(0, Tm, J)
for (t in 1:Tm) {
  draws <- matrix(rnorm(n_cons*3), n_cons, 3) * 0.4 + matrix(b, n_cons, 3, byrow=TRUE)
  util <- draws %*% t(X[t,,]) + matrix(xi[t,], n_cons, J, byrow=TRUE)
  util <- cbind(util, 0)
  p <- exp(util - apply(util,1,max))
  p <- p / rowSums(p)
  shares[t,] <- colMeans(p[,1:J])
}
# Contração simplificada
delta <- log(shares + 1e-12) - log(1 - rowSums(shares) + 1e-12)
cat("BLP (R) – Cap. 14, pp. 127-138\nAutor: Luiz Tiago Wilcke\n")
cat("Delta médio por produto:", round(colMeans(delta),3), "\n")
cat("Shares médios:", round(colMeans(shares),3), "\n")
