library(data.table)   
library(dplyr)        
library(lubridate)    
library(zoo)  

# List all CSV files in the directory
csv_files <- list.files(
  path = "/Users/andrescadena/Library/CloudStorage/OneDrive-europa-uni.de/Deep-NN/Project Task and Files-20250508/data_rnn", 
  pattern = "\\.csv$", 
  full.names = TRUE)

# Read and combine all CSV files into one data.table
combined_data <- rbindlist(
  lapply(csv_files, fread), #fread: Similar to read.table but faster and more convenient. All controls such as sep, colClasses and nrows are automatically detected.
  fill = TRUE, 
  use.names = TRUE)

filtered_data <- combined_data[combined_data$MapCode == "DE_LU", ]

# Convert datetime to POSIXct (ensure UTC timezone)
filtered_data$DateTime <- as.POSIXct(
  filtered_data$DateTime, 
  tz = "UTC"
)

# Define start and end dates
start_date <- as.POSIXct("2018-11-01 00:00:00", tz = "UTC")
end_date <- as.POSIXct("2023-11-30 23:00:00", tz = "UTC")

# Generate complete hourly sequence
full_time_seq <- data.frame(
  DateTime = seq(start_date, end_date, by = "hour")
)

# Merge with complete time sequence
complete_data <- full_time_seq %>% 
  left_join(filtered_data, by = "DateTime") %>% 
  arrange(DateTime)

# Interpolate missing prices linearly
complete_data$Price <- na.approx(complete_data$Price)

# Fill leading/trailing NAs (if any remain)
complete_data$Price <- na.locf(complete_data$Price, na.rm = FALSE)
complete_data$Price <- na.locf(complete_data$Price, fromLast = TRUE, na.rm = FALSE) #To ensure start and end do not have NAs

# Remove duplicate timestamps (keep first occurrence)
complete_data <- complete_data %>% 
  distinct(DateTime, .keep_all = TRUE)

# Verify no missing values
sum(is.na(complete_data$Price))  # Should return 0

#________________________________________________________________________________

lookback <- 24

price_cap <- quantile(complete_data$Price, 0.995, na.rm = TRUE)
complete_data$Price[complete_data$Price > price_cap] <- price_cap # replacing extreme prices with a more reasonable ceiling.

data <- complete_data$Price 

X <- matrix(nrow = length(data) - lookback, ncol = lookback)
Y <- c()

for (i in 1:(length(data) - lookback)) {
  X[i,] <- data[i:(i + lookback - 1)]
  Y[i] <- data[i + lookback]
}

train_size <- floor(0.8 * nrow(X))
x_train <- X[1:train_size, ]
y_train <- Y[1:train_size]
x_test <- X[(train_size + 1):nrow(X), ]
y_test <- Y[(train_size + 1):length(Y)]

standardized_data <- X.standardize(x_train, x_test)
x_train <- standardized_data[[1]]
x_test <- standardized_data[[2]]

y_mean <- mean(y_train)
y_sd <- sd(y_train)
y_train_scaled <- (y_train - y_mean) / y_sd
y_test_scaled  <- (y_test - y_mean) / y_sd

plot(y_train_scaled, type = "l", main = "Target over Time (Standardized)", ylab = "Scaled Price")
abline(h = 0, col = "red", lty = 2)


#________________________________________________________________________________

# RNN implementation using a simple architecture (many to one RNN)

# Xavier initialization for tanh activation
rnn.weights.init <- function(input_dim, hidden_dim, output_dim) {
  xavier_limit_xh <- sqrt(6 / (input_dim + hidden_dim))
  W_xh <- matrix(runif(hidden_dim * input_dim, min = -xavier_limit_xh, max = xavier_limit_xh),
                 nrow = hidden_dim, ncol = input_dim)
  
  xavier_limit_hh <- sqrt(6 / (hidden_dim + hidden_dim))
  W_hh <- matrix(runif(hidden_dim * hidden_dim, min = -xavier_limit_hh, max = xavier_limit_hh),
                 nrow = hidden_dim, ncol = hidden_dim)
  
  xavier_limit_hy <- sqrt(6 / (hidden_dim + output_dim))
  W_hy <- matrix(runif(output_dim * hidden_dim, min = -xavier_limit_hy, max = xavier_limit_hy),
                 nrow = output_dim, ncol = hidden_dim)
  
  b_h <- matrix(0, nrow = hidden_dim, ncol = 1)
  b_y <- matrix(0, nrow = output_dim, ncol = 1)
  
  list(W_xh = W_xh, W_hh = W_hh, W_hy = W_hy, b_h = b_h, b_y = b_y)
}

rnn.weights.pack <- function(weights_list) {
  c(as.vector(weights_list$W_xh),
    as.vector(weights_list$W_hh),
    as.vector(weights_list$W_hy),
    as.vector(weights_list$b_h),
    as.vector(weights_list$b_y))
}

rnn.weights.unpack <- function(weights_vector, input_dim, hidden_dim, output_dim) {
  idx <- 0
  
  len_W_xh <- hidden_dim * input_dim
  W_xh <- matrix(weights_vector[(idx + 1):(idx + len_W_xh)], nrow = hidden_dim, ncol = input_dim)
  idx <- idx + len_W_xh
  
  len_W_hh <- hidden_dim * hidden_dim
  W_hh <- matrix(weights_vector[(idx + 1):(idx + len_W_hh)], nrow = hidden_dim, ncol = hidden_dim)
  idx <- idx + len_W_hh
  
  len_W_hy <- output_dim * hidden_dim
  W_hy <- matrix(weights_vector[(idx + 1):(idx + len_W_hy)], nrow = output_dim, ncol = hidden_dim)
  idx <- idx + len_W_hy
  
  len_b_h <- hidden_dim
  b_h <- matrix(weights_vector[(idx + 1):(idx + len_b_h)], nrow = hidden_dim, ncol = 1)
  idx <- idx + len_b_h
  
  len_b_y <- output_dim
  b_y <- matrix(weights_vector[(idx + 1):(idx + len_b_y)], nrow = output_dim, ncol = 1)
  
  list(W_xh = W_xh, W_hh = W_hh, W_hy = W_hy, b_h = b_h, b_y = b_y)
}


# Tanh activation
rnn_forward <- function(x_sequence, weights, h_prev = NULL) {
  W_xh <- weights$W_xh
  W_hh <- weights$W_hh
  W_hy <- weights$W_hy
  b_h <- weights$b_h
  b_y <- weights$b_y
  
  hidden_dim <- nrow(W_hh)
  input_dim <- ncol(W_xh)
  sequence_length <- length(x_sequence)
  
  if (is.null(h_prev)) {
    h_prev <- matrix(0, nrow = hidden_dim, ncol = 1)
  }
  
  hidden_states <- vector("list", sequence_length)
  input_values_at_t <- vector("list", sequence_length)
  net_inputs_at_t <- vector("list", sequence_length)
  
  for (t in 1:sequence_length) {
    x_t <- matrix(x_sequence[t], nrow = input_dim, ncol = 1)
    input_values_at_t[[t]] <- x_t
    
    net_h <- W_xh %*% x_t + W_hh %*% h_prev + b_h
    net_inputs_at_t[[t]] <- net_h
    
    h_t <- tanh(net_h)
    
    hidden_states[[t]] <- h_t
    h_prev <- h_t
  }
  
  net_o <- W_hy %*% h_prev + b_y
  y_hat <- as.numeric(net_o)
  
  return(list(
    y_hat = y_hat,
    hidden_states = hidden_states,
    input_values_at_t = input_values_at_t,
    net_inputs_at_t = net_inputs_at_t
  ))
}

rnn_bptt <- function(y_true, forward_result, weights) {
  net_inputs <- forward_result$net_inputs_at_t
  y_hat <- forward_result$y_hat
  H <- forward_result$hidden_states
  X <- forward_result$input_values_at_t
  
  W_xh <- weights$W_xh
  W_hh <- weights$W_hh
  W_hy <- weights$W_hy
  
  hidden_dim <- nrow(W_hh)
  input_dim <- ncol(W_xh)
  output_dim <- nrow(W_hy)
  T_seq_len <- length(H)
  
  dW_xh <- matrix(0, nrow = hidden_dim, ncol = input_dim)
  dW_hh <- matrix(0, nrow = hidden_dim, ncol = hidden_dim)
  dW_hy <- matrix(0, nrow = output_dim, ncol = hidden_dim)
  db_h <- matrix(0, nrow = hidden_dim, ncol = 1)
  db_y <- matrix(0, nrow = output_dim, ncol = 1)
  
  delta_o <- y_hat - y_true
  
  h_T <- H[[T_seq_len]]
  dW_hy <- delta_o %*% t(h_T)
  db_y <- delta_o
  
  dL_dh_from_next <- matrix(0, nrow = hidden_dim, ncol = 1)
  
  for (t in T_seq_len:1) {
    h_t <- H[[t]]
    x_t <- X[[t]]
    h_prev_for_t <- if (t == 1) matrix(0, nrow = hidden_dim, ncol = 1) else H[[t-1]]
    
    dL_dh_t_total <- dL_dh_from_next
    if (t == T_seq_len) {
      dL_dh_t_total <- dL_dh_t_total + (t(W_hy) %*% delta_o)
    }
    
    net_h <- net_inputs[[t]]
    tanh_derivative <- 1 - (tanh(net_h))^2
    delta_net_h <- dL_dh_t_total * tanh_derivative
    
    dW_xh <- dW_xh + delta_net_h %*% t(x_t)
    dW_hh <- dW_hh + delta_net_h %*% t(h_prev_for_t)
    db_h <- db_h + delta_net_h
    
    dL_dh_from_next <- t(W_hh) %*% delta_net_h
  }
  
  return(list(dW_xh = dW_xh, dW_hh = dW_hh, dW_hy = dW_hy, db_h = db_h, db_y = db_y))
}


rnn_update_weights <- function(weights, grads, learning_rate) {
  weights$W_xh <- weights$W_xh - learning_rate * grads$dW_xh
  weights$W_hh <- weights$W_hh - learning_rate * grads$dW_hh
  weights$W_hy <- weights$W_hy - learning_rate * grads$dW_hy
  weights$b_h <- weights$b_h - learning_rate * grads$db_h
  weights$b_y <- weights$b_y - learning_rate * grads$db_y
  return(weights)
}

rnn.cost.derivative.batch <- function(flat_weights, y, X, nn, hid_n, out_n, input_dim, h_initial_overall = NULL) {
  
  # Unpack flat weights into matrices/vectors
  weights <- rnn.weights.unpack(flat_weights, input_dim, hid_n, out_n)
  
  num_samples <- nrow(X)
  
  # Initialize total gradients as zero matrices/vectors
  total_dW_xh <- matrix(0, nrow = hid_n, ncol = input_dim)
  total_dW_hh <- matrix(0, nrow = hid_n, ncol = hid_n)
  total_dW_hy <- matrix(0, nrow = out_n, ncol = hid_n)
  total_db_h <- matrix(0, nrow = hid_n, ncol = 1)
  total_db_y <- matrix(0, nrow = out_n, ncol = 1)
  
  for (i in 1:num_samples) {
    x_seq <- X[i, ]
    y_true_val <- y[i]
    
    # Forward pass for sample i
    forward_out <- rnn_forward(x_seq, weights, h_prev = h_initial_overall)
    
    # Backpropagation for sample i
    grads <- rnn_bptt(y_true_val, forward_out, weights)
    
    # Accumulate gradients
    total_dW_xh <- total_dW_xh + grads$dW_xh
    total_dW_hh <- total_dW_hh + grads$dW_hh
    total_dW_hy <- total_dW_hy + grads$dW_hy
    total_db_h <- total_db_h + grads$db_h
    total_db_y <- total_db_y + grads$db_y
  }
  
  # Pack total gradients into flat vector (to match flat_weights shape)
  packed_gradients <- rnn.weights.pack(list(
    W_xh = total_dW_xh,
    W_hh = total_dW_hh,
    W_hy = total_dW_hy,
    b_h = total_db_h,
    b_y = total_db_y
  ))
  
  return(packed_gradients)
}

rnn_predict_for_bgd_wrapper <- function(flat_weights, X, hid_n, out_n, input_dim, h_initial_overall = NULL) {
  
  # Unpack weights vector into matrices
  weights <- rnn.weights.unpack(flat_weights, input_dim, hid_n, out_n)
  
  num_samples <- nrow(X)
  y_preds <- numeric(num_samples)
  
  # Loop over all sequences in X
  for (i in 1:num_samples) {
    # Run forward pass for each input sequence
    forward_out <- rnn_forward(X[i, ], weights, h_prev = h_initial_overall)
    y_preds[i] <- forward_out$y_hat
  }
  
  # Return list containing predictions vector
  return(list(y_est = y_preds))
}


input_dim <- 1  
hidden_dim <- 16
output_dim <- 1

set.seed(1234)
initial_weights_list <- rnn.weights.init(input_dim, hidden_dim, output_dim)
trained_weights <- rnn.weights.pack(initial_weights_list)

trained_weights <- bgd(
  weights = initial_weights_flat,
  y = y_train_scaled,
  X = x_train,
  hid_n = hidden_dim,
  out_n = output_dim,
  nn = rnn_predict_for_bgd_wrapper,
  costderiv = rnn.cost.derivative.batch,
  epochs = 100,
  lr = 0.00001,
  input_dim = input_dim
)

#_______________________________________________________________________________

preds_scaled <- rnn_predict_for_bgd_wrapper(
  trained_weights, x_test, hid_n = 16, out_n = 1, input_dim = 1
)$y_est

preds <- preds_scaled * y_sd + y_mean  # inverse standardization
actuals <- y_test

plot(actuals, type = "l", col = "black")
lines(preds, col = "red")
legend("topright", legend = c("Actual", "Predicted"), col = c("black", "red"), lty = 1)

subset_range <- 1:500
plot(actuals[subset_range], type = "l", col = "black", lwd = 2,
     main = "Actual Prices (First 500)", xlab = "Time", ylab = "Price")

plot(preds[subset_range], type = "l", col = "red", lwd = 2,
     main = "Predicted Prices (First 500)", xlab = "Time", ylab = "Price")


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

mse <- function(actual, pred) mean((actual - pred)^2)

train_mse <- mse(y_train, train_preds)
test_mse  <- mse(y_test, test_preds)

cat("Train MSE:", train_mse, "\n")
cat("Test MSE:", test_mse, "\n")



