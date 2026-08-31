# Cox – Cap. 26, pp. 248-256 | Autor: Luiz Tiago Wilcke
library(survival)
set.seed(42)
n <- 500
X <- matrix(rnorm(n*2), n, 2)
lam <- exp(X %*% c(0.8, -0.5))
tempo <- rweibull(n, 1.5) / as.vector(lam)
cens <- runif(n, 0, max(tempo)*0.8)
evento <- as.numeric(tempo <= cens)
t_obs <- pmin(tempo, cens)
fit <- coxph(Surv(t_obs, evento) ~ X)
cat("COX (R) – Cap. 26, pp. 248-256\nAutor: Luiz Tiago Wilcke\n")
print(summary(fit))
