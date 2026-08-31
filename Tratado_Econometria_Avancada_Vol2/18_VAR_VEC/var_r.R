# Modelo 18 – VAR / VEC
# Autor: Luiz Tiago Wilcke
library(vars)
set.seed(42)
T <- 200
e <- rnorm(T); u <- rnorm(T)
x <- cumsum(e); y <- 0.5*x + u
dados <- data.frame(y=y, x=x)
var_mod <- VAR(dados, p=2, type="const")
cat("VAR(2) (R)\nAutor: Luiz Tiago Wilcke\n")
print(summary(var_mod))
# Johansen
library(urca)
joh <- ca.jo(dados, type="trace", K=2, spec="longrun")
print(summary(joh))
