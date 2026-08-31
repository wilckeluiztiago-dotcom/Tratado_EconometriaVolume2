# Cópulas – Cap. 19, pp. 183-184 | Autor: Luiz Tiago Wilcke
set.seed(42)
n <- 1000; theta <- 2.5
u <- runif(n); v <- runif(n)
w <- ((v^(-theta/(1+theta)) - 1) * u^(-theta) + 1)^(-1/theta)
x <- qt(u, 5); y <- qt(w, 5)
cat("CÓPULAS (R) – Cap. 19, pp. 183-184\nAutor: Luiz Tiago Wilcke\n")
cat("Correlação:", cor(x,y), "\n")
cat("Use copula/VineCopula para ajuste completo.\n")
