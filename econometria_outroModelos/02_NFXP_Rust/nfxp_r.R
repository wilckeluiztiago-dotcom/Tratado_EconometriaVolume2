# NFXP Rust – Cap. 15, pp. 139-150 | Autor: Luiz Tiago Wilcke
set.seed(42)
n_states <- 15; beta <- 0.95
theta_true <- c(0.5, 4.0)
# EV e simulação simplificados
cat("NFXP (R) – Cap. 15, pp. 139-150\nAutor: Luiz Tiago Wilcke\n")
cat("Modelo de substituição dinâmica (Rust 1987).\n")
cat("Theta verdadeiro (c_manut, RC):", theta_true, "\n")
cat("Use o script Python para estimação completa por nested fixed point.\n")
