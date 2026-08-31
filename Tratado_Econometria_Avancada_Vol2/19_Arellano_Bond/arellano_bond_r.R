# Modelo 19 – Arellano-Bond
# Autor: Luiz Tiago Wilcke
library(plm)
set.seed(42)
N <- 100; T <- 8
a <- rnorm(N)
x <- matrix(rnorm(N*T), N, T)
y <- matrix(0, N, T)
y[,1] <- a + rnorm(N)
for (t in 2:T) y[,t] <- 0.6*y[,t-1] + 1.0*x[,t] + a + rnorm(N)
dados <- data.frame(
  id = rep(1:N, each=T),
  tempo = rep(1:T, times=N),
  y = as.vector(t(y)),
  x = as.vector(t(x))
)
pdados <- pdata.frame(dados, index=c("id","tempo"))
# Diferenças OLS (aproximação)
mod <- plm(y ~ lag(y,1) + x, data=pdados, model="fd")
cat("ARELLANO-BOND aproximado (R / plm fd)\nAutor: Luiz Tiago Wilcke\n")
print(summary(mod))
cat("Verdadeiro: rho=0.6, beta=1.0\n")
