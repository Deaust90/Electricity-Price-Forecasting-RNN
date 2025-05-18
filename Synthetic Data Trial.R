
bgd <- function(weights, y, X, hid_n, out_n, nn, costderiv, 
                epochs = 3000, lr = 0.00001, ...) {
  
  for (i in 1:epochs) {
    deriv <- costderiv(weights, y, X, nn, hid_n, out_n, ...)
    weights <- weights - lr * deriv
    
    # Every 10 epochs, print loss
    if (i %% 10 == 0) {
      preds <- nn(weights, X, hid_n, out_n, ...)
      mse <- mean((preds$y_est - y)^2)
      cat(sprintf("Epoch %d: MSE = %.6f\n", i, mse))
    }
  }
  
  return(weights)
}


#_______________________________________________________________________________
small_x_train <- x_train[1:1000, ]
small_y_train <- y_train[1:1000]

mean_x <- mean(small_x_train)
sd_x <- sd(small_x_train)
small_x_train_scaled <- (small_x_train - mean_x) / sd_x

# Standardize targets
mean_y <- mean(small_y_train)
sd_y <- sd(small_y_train)
small_y_train_scaled <- (small_y_train - mean_y) / sd_y

trained_weights <- bgd(
  weights = initial_weights_flat,
  y = small_y_train_scaled,
  X = small_x_train_scaled,
  hid_n = 16,
  out_n = 1,
  nn = rnn_predict_for_bgd_wrapper,
  costderiv = rnn.cost.derivative.batch,
  epochs = 200,
  lr = 0.0001,
  input_dim = 1
)

plot(trained_weights$mse_per_epoch, type = "l", col = "blue",
     main = "Training Loss (MSE) Over Epochs",
     xlab = "Epoch", ylab = "MSE")

summary(x_train)
summary(y_train_scaled)

any(is.infinite(x_train))
any(is.infinite(y_train_scaled))
any(is.na(x_train))
any(is.na(y_train_scaled))

weights <- rnn.weights.init(1, 16, 1)
summary(as.vector(weights$W_hy))  # Inspect output layer weights

#_______________________________________________________________________________
train_losses <- c()
val_losses <- c()

for (epoch in 1:100) {
  
  # 1) Update weights by performing one BGD step here
  trained_weights <- bgd(
    weights = trained_weights,
    y = y_train_scaled,
    X = x_train,
    hid_n = hidden_dim,
    out_n = output_dim,
    nn = rnn_predict_for_bgd_wrapper,
    costderiv = rnn.cost.derivative.batch,
    epochs = 1,
    lr = 0.00001,
    input_dim = input_dim
  )
  
  # 2) Predict on train set with current weights
  train_preds_scaled <- rnn_predict_for_bgd_wrapper(
    flat_weights = trained_weights,
    X = x_train,
    hid_n = 16,
    out_n = 1,
    input_dim = 1
  )$y_est
  
  # 3) Calculate training loss (MSE)
  train_loss <- mean((train_preds_scaled - y_train_scaled)^2)
  train_losses <- c(train_losses, train_loss)
  
  # 4) Predict on validation/test set with current weights
  val_preds_scaled <- rnn_predict_for_bgd_wrapper(
    flat_weights = trained_weights,
    X = x_test,
    hid_n = 16,
    out_n = 1,
    input_dim = 1
  )$y_est
  
  # 5) Calculate validation loss (MSE)
  val_loss <- mean((val_preds_scaled - y_test_scaled)^2)
  val_losses <- c(val_losses, val_loss)
  
  # Optionally print progress every some epochs
  if (epoch %% 10 == 0) {
    cat(sprintf("Epoch %d: Train Loss = %.4f | Val Loss = %.4f\n", epoch, train_loss, val_loss))
  }
}


plot(train_losses, type = "l", col = "blue", lwd = 2,
     ylim = range(c(train_losses, val_losses)),
     xlab = "Epoch", ylab = "Loss (MSE)",
     main = "Training and Validation Loss Over Epochs")
lines(val_losses, col = "red", lwd = 2)
legend("topright", legend = c("Train Loss", "Validation Loss"), 
       col = c("blue", "red"), lwd = 2)

#_______________________________________________________________________________

# Predict on training set
train_preds_scaled <- rnn_predict_for_bgd_wrapper(
  flat_weights = trained_weights,
  X = x_train,
  hid_n = 16,
  out_n = 1,
  input_dim = 1
)$y_est

# Predict on test set
test_preds_scaled <- rnn_predict_for_bgd_wrapper(
  flat_weights = trained_weights,
  X = x_test,
  hid_n = 16,
  out_n = 1,
  input_dim = 1
)$y_est

# Undo standardization (if y was standardized)
train_preds <- train_preds_scaled * y_sd + y_mean
test_preds  <- test_preds_scaled  * y_sd + y_mean

par(mfrow = c(1, 2))  # 1 row, 2 columns

plot(y_train, type = "l", col = "black", main = "Train: Actual vs Predicted",
     xlab = "Time", ylab = "Price")
lines(train_preds, col = "red")

plot(y_test, type = "l", col = "black", main = "Test: Actual vs Predicted",
     xlab = "Time", ylab = "Price")
lines(test_preds, col = "red")

plot(c(y_train, y_test), type = "l", col = "black", lwd = 2,
     main = "Train and Test Prediction Comparison",
     xlab = "Time", ylab = "Price")
lines(c(train_preds, test_preds), col = "red", lwd = 2)
abline(v = length(y_train), lty = 2, col = "blue")  # Mark train/test boundary
legend("topright", legend = c("Actual", "Predicted"), col = c("black", "red"), lwd = 2)

train_mse <- mse(y_train, train_preds)
test_mse  <- mse(y_test, test_preds)

cat("Train MSE:", train_mse, "\n")
cat("Test MSE:", test_mse, "\n")
