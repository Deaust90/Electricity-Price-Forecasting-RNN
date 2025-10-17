# Forward pass and loss for Wx
fwd_Wx <- function(Wx, H_prev, X_t, Y_t, Wh, Wo) {
  Z_t <- Wh * H_prev + Wx * X_t
  H_t <- tanh(Z_t)
  Y_pred <- Wo * H_t
  L <- 0.5 * (Y_pred - Y_t)^2
  return(as.numeric(L))
}

# Analytical gradient dL/dWx
analytical_grad_Wx <- function(Wx, H_prev, X_t, Y_t, Wh, Wo) {
  Z_t <- Wh * H_prev + Wx * X_t
  H_t <- tanh(Z_t)
  Y_pred <- Wo * H_t
  
  dL_dYpred <- Y_pred - Y_t
  dYpred_dHt <- Wo
  dHt_dZt <- 1 - tanh(Z_t)^2
  dZt_dWx <- X_t
  
  return(dL_dYpred * dYpred_dHt * dHt_dZt * dZt_dWx)
}

# Forward pass and loss for Wo
fwd_Wo <- function(Wo, H_prev, X_t, Y_t, Wh, Wx) {
  Z_t <- Wh * H_prev + Wx * X_t
  H_t <- tanh(Z_t)
  Y_pred <- Wo * H_t
  L <- 0.5 * (Y_pred - Y_t)^2
  return(as.numeric(L))
}

# Analytical gradient dL/dWo
analytical_grad_Wo <- function(Wo, H_prev, X_t, Y_t, Wh, Wx) {
  Z_t <- Wh * H_prev + Wx * X_t
  H_t <- tanh(Z_t)
  Y_pred <- Wo * H_t
  
  dL_dYpred <- Y_pred - Y_t
  dYpred_dWo <- H_t
  
  return(dL_dYpred * dYpred_dWo)
}

# Numerical gradient via finite differences
numerical_grad <- function(param, f, eps = 1e-5) {
  f_plus  <- f(param + eps)
  f_minus <- f(param - eps)
  return((f_plus - f_minus) / (2 * eps))
}

# Example fixed values
Wh <- 0.5; Wx <- 0.3; Wo <- 0.7
H_prev <- 0.1; X_t <- 0.2; Y_t <- 0.05

# Check Wx gradient
num_grad_Wx <- numerical_grad(Wx, function(w) fwd_Wx(w, H_prev, X_t, Y_t, Wh, Wo))
ana_grad_Wx <- analytical_grad_Wx(Wx, H_prev, X_t, Y_t, Wh, Wo)
cat("Wx numerical gradient:", num_grad_Wx, "\n")
cat("Wx analytical gradient:", ana_grad_Wx, "\n\n")

# Check Wo gradient
num_grad_Wo <- numerical_grad(Wo, function(w) fwd_Wo(w, H_prev, X_t, Y_t, Wh, Wx))
ana_grad_Wo <- analytical_grad_Wo(Wo, H_prev, X_t, Y_t, Wh, Wx)
cat("Wo numerical gradient:", num_grad_Wo, "\n")
cat("Wo analytical gradient:", ana_grad_Wo, "\n")

