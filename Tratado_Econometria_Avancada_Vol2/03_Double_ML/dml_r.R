# ============================================================
# Modelo 03 – Double Machine Learning (DML)
# Tratado de Econometria Avançada – Volume II
# Autor: Luiz Tiago Wilcke
# ============================================================

set.seed(42)
n <- 2000

# Dados sintéticos
X <- matrix(rnorm(n * 10), n, 10)
colnames(X) <- paste0("covariavel_", 1:10)
tratamento <- as.numeric(0.5*X[,1] + 0.3*X[,2] + rnorm(n) > 0)
efeito_verdadeiro <- 2.0
resultado <- efeito_verdadeiro * tratamento + 1.5*X[,1] + 0.8*X[,3]^2 + rnorm(n)

dados <- data.frame(resultado = resultado, tratamento = tratamento, X)

# DML simplificado com residualização via randomForest
# install.packages(c("randomForest", "AER"))
library(randomForest)

n_folds <- 5
folds <- sample(rep(1:n_folds, length.out = n))
residuos_Y <- numeric(n)
residuos_D <- numeric(n)

for (k in 1:n_folds) {
  treino <- folds != k
  teste  <- folds == k
  # Nuisance functions
  rf_g <- randomForest(resultado ~ ., data = dados[treino, -2], ntree = 100)
  rf_m <- randomForest(tratamento ~ ., data = dados[treino, -1], ntree = 100)
  residuos_Y[teste] <- dados$resultado[teste] - predict(rf_g, dados[teste, ])
  residuos_D[teste] <- dados$tratamento[teste] - predict(rf_m, dados[teste, ])
}

theta <- sum(residuos_D * residuos_Y) / sum(residuos_D^2)
psi <- residuos_D * (residuos_Y - theta * residuos_D)
se <- sqrt(mean(psi^2) / n) / mean(residuos_D^2)

cat("============================================================\n")
cat("DOUBLE MACHINE LEARNING (DML) – R\n")
cat("Autor: Luiz Tiago Wilcke\n")
cat("============================================================\n")
cat("Efeito causal estimado (theta):", round(theta, 4), "\n")
cat("Erro-padrão:", round(se, 4), "\n")
cat("IC 95%: [", round(theta - 1.96*se, 4), ",", round(theta + 1.96*se, 4), "]\n")
cat("Efeito verdadeiro:", efeito_verdadeiro, "\n")
