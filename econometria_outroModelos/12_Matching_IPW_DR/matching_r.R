# Matching/IPW/DR – Cap. 32, pp. 306-314 | Autor: Luiz Tiago Wilcke
set.seed(42)
n <- 1000
X <- matrix(rnorm(n*3), n, 3)
ps <- 1/(1+exp(-(0.5*X[,1]-0.3*X[,2])))
D <- rbinom(n,1,ps)
Y <- 1 + 2*D + 0.8*X[,1] + rnorm(n)
cat("MATCHING/IPW/DR (R) – Cap. 32, pp. 306-314\nAutor: Luiz Tiago Wilcke\n")
cat("ATE verdadeiro: 2.0\n")
