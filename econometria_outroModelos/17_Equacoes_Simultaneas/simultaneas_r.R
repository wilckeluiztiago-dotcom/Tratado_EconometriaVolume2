# Equações Simultâneas – Cap. 23, pp. 222-230 | Autor: Luiz Tiago Wilcke
library(AER)
set.seed(42)
n <- 500
z1 <- rnorm(n); z2 <- rnorm(n)
e1 <- rnorm(n); e2 <- 0.5*e1 + rnorm(n)
A <- matrix(c(1, -0.4, -0.6, 1), 2, 2)
Y <- t(solve(A, rbind(1*z1+e1, 1.2*z2+e2)))
y1 <- Y[,1]; y2 <- Y[,2]
fit <- ivreg(y1 ~ y2 + z1 | z1 + z2)
cat("2SLS (R) – Cap. 23, pp. 222-230\nAutor: Luiz Tiago Wilcke\n")
print(summary(fit))
