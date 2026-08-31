# Poisson/ZIP – Cap. 28, pp. 267-275 | Autor: Luiz Tiago Wilcke
set.seed(42)
n <- 800
X <- cbind(1, rnorm(n), rbinom(n,1,0.4))
mu <- exp(X %*% c(0.5, 0.4, -0.3))
y <- ifelse(runif(n) < 0.25, 0, rpois(n, mu))
cat("POISSON/ZIP (R) – Cap. 28, pp. 267-275\nAutor: Luiz Tiago Wilcke\n")
cat("% zeros:", mean(y==0), " média:", mean(y), "\n")
print(summary(glm(y ~ X[,-1], family=poisson)))
