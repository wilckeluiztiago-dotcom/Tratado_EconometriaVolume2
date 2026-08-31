# GPV Leilões – Cap. 18, pp. 173-180 | Autor: Luiz Tiago Wilcke
set.seed(42)
n_b <- 3; n_a <- 800
v <- matrix(runif(n_a*n_b), n_a, n_b)
b <- (n_b-1)/n_b * v
cat("GPV LEILÕES (R) – Cap. 18, pp. 173-180\nAutor: Luiz Tiago Wilcke\n")
cat("Lance médio:", mean(b), " Valor médio:", mean(v), "\n")
