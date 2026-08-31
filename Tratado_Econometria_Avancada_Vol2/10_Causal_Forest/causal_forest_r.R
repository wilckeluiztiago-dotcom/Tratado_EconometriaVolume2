# Modelo 10 – Causal Forest
# Autor: Luiz Tiago Wilcke
set.seed(42)
n <- 1000
X <- matrix(rnorm(n*5), n, 5)
tratamento <- as.numeric(X[,1] + rnorm(n) > 0)
efeito_v <- 1 + 0.5*X[,2]
y <- efeito_v * tratamento + X[,3] + rnorm(n)
# T-learner com randomForest
library(randomForest)
rf0 <- randomForest(X[tratamento==0,], y[tratamento==0], ntree=100)
rf1 <- randomForest(X[tratamento==1,], y[tratamento==1], ntree=100)
cate <- predict(rf1, X) - predict(rf0, X)
cat("CAUSAL FOREST (R)\nAutor: Luiz Tiago Wilcke\n")
cat("CATE médio estimado:", mean(cate), "\n")
cat("CATE médio verdadeiro:", mean(efeito_v), "\n")
