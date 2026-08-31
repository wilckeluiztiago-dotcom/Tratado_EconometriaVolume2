# Modelo 06 – STAR
# Autor: Luiz Tiago Wilcke
library(tsDyn)
set.seed(42)
n <- 300
y <- numeric(n)
for (t in 2:n) {
  G <- 1/(1+exp(-2*(y[t-1]-0)))
  y[t] <- (0.4+0.5*y[t-1])*(1-G) + (-0.1+0.2*y[t-1])*G + rnorm(1,0,0.7)
}
mod <- star(y, m=1, thDelay=1, control=list(maxit=100))
cat("MODELO STAR (R)\nAutor: Luiz Tiago Wilcke\n")
print(summary(mod))
