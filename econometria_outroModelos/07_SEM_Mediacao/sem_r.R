# SEM Mediação – Cap. 21, pp. 202-211 | Autor: Luiz Tiago Wilcke
set.seed(42)
n <- 800
X <- rnorm(n); M <- 0.6*X + rnorm(n); Y <- 0.3*X + 0.5*M + rnorm(n)
a <- cov(X,M)/var(X)
fit <- lm(Y ~ X + M)
b <- coef(fit)[3]; cp <- coef(fit)[2]
cat("SEM/MEDIAÇÃO (R) – Cap. 21, pp. 202-211\nAutor: Luiz Tiago Wilcke\n")
cat("Indireto:", a*b, " Direto:", cp, " Total:", cp+a*b, "\n")
