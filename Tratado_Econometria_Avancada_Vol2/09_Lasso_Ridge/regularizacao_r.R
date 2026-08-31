# Modelo 09 – Regularização
# Autor: Luiz Tiago Wilcke
library(glmnet)
set.seed(42)
n <- 200; p <- 50
X <- matrix(rnorm(n*p), n, p)
beta <- c(1.5,-1.2,0.8,0.5,-0.7, rep(0,45))
y <- X %*% beta + rnorm(n)
cv_lasso <- cv.glmnet(X, y, alpha=1)
cv_ridge <- cv.glmnet(X, y, alpha=0)
cat("REGULARIZAÇÃO (R / glmnet)\nAutor: Luiz Tiago Wilcke\n")
cat("Lasso lambda.min:", cv_lasso$lambda.min, "\n")
print(coef(cv_lasso, s="lambda.min")[1:6,])
