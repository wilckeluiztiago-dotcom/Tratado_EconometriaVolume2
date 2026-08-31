# Tobit/Heckman – Cap. 27, pp. 257-265 | Autor: Luiz Tiago Wilcke
set.seed(42)
n <- 1000
X <- cbind(1, rnorm(n))
y <- pmax(X %*% c(1, 1.5) + rnorm(n), 0)
cat("TOBIT/HECKMAN (R) – Cap. 27, pp. 257-265\nAutor: Luiz Tiago Wilcke\n")
cat("Proporção censurada:", mean(y==0), "\n")
# library(AER); tobit(y ~ X[,-1])
