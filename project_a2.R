library(data.table)   
library(dplyr)        
library(lubridate)    
library(zoo)
library(tensorflow)
library(keras)
library(tidyverse)

# 1. Data Preprocessing ----------------------------------
# 2. Prepare Data for the RNN ----------------------------
# Same as project_a1, the RNN in base R

## 3.1 TensorFlow Model Definition and Manual Weight Setting ----------------

input_dim_R <- 1    
hidden_dim_R <- 16  
output_dim_R <- 1
seed_R <- 1234

set.seed(seed_R)

# Calling rnn.weights.init function from proejct_a1:
initial_weights_list_R <- rnn.weights.init(
  input_dim = input_dim_R,
  hidden_dim = hidden_dim_R,
  output_dim = output_dim_R
)


# Hyperparameters for TensorFlow model (matching R)
tf_input_timesteps <- 24
tf_input_features  <- 1
tf_hidden_units    <- 16
tf_output_units    <- 1

# Model buidling without initilization
model_tf_manual_weights <- keras_model_sequential(name = "tf_rnn_manual_weights")
model_tf_manual_weights %>%
  layer_simple_rnn(
    units = tf_hidden_units,
    activation = 'tanh',
    input_shape = c(tf_input_timesteps, tf_input_features),
    use_bias = TRUE,
    name = "simple_rnn_layer_manual" # No initializers specified; we set weights manually
  ) %>%
  layer_dense(
    units = tf_output_units,
    activation = 'linear',
    use_bias = TRUE,
    name = "output_layer_manual"
  )

# --- Note: project_a1 needs to be added to the environment --

# Prepare R weights for TensorFlow Keras format

# kernel: x→h
tf_W_xh_from_R <- array_reshape(
  t(initial_weights_list_R$W_xh),                      # transpose to (1 × 16)
  dim = c(tf_input_features, tf_hidden_units)          # 1 × 16
)

# recurrent_kernel: h→h 
tf_W_hh_from_R <- array_reshape(
  initial_weights_list_R$W_hh,                         # already 16 × 16 in R
  dim = c(tf_hidden_units, tf_hidden_units)            # 16 × 16
)

# bias (hidden)
tf_b_h_from_R <- as.numeric(initial_weights_list_R$b_h) # bias in R (16 zeros) travels intact into TensorFlow
if (length(tf_b_h_from_R) == 1)                           
  tf_b_h_from_R <- rep(tf_b_h_from_R, tf_hidden_units)    
tf_b_h_from_R <- array_reshape(tf_b_h_from_R, dim = c(tf_hidden_units))

# kernel: h→y
tf_W_hy_from_R <- array_reshape(
  t(initial_weights_list_R$W_hy),                      # transpose to 16 × 1
  dim = c(tf_hidden_units, tf_output_units)            # 16 × 1
)

# bias (output)
tf_b_y_from_R <- array_reshape(
  as.numeric(initial_weights_list_R$b_y),              # make 1-vector
  dim = c(tf_output_units)                             # length 1
)

# Build the model by calling it with a dummy input: to trigger model building before manually setting weights
dummy_input_shape <- c(1L, as.integer(tf_input_timesteps), as.integer(tf_input_features))
dummy_input <- tf$zeros(shape = dummy_input_shape, dtype = tf$float32)
invisible(model_tf_manual_weights(dummy_input)) # Builds the model layers

# Set the prepared R weights into the TensorFlow Model
rnn_layer_tf <- get_layer(model_tf_manual_weights, name = "simple_rnn_layer_manual")
set_weights(rnn_layer_tf, list(tf_W_xh_from_R, tf_W_hh_from_R, tf_b_h_from_R))

output_layer_tf <- get_layer(model_tf_manual_weights, name = "output_layer_manual")
set_weights(output_layer_tf, list(tf_W_hy_from_R, tf_b_y_from_R))

print("TensorFlow model defined and weights manually set from R initial weights.")
summary(model_tf_manual_weights)


## 3.2 Reshape Input Data for TensorFlow --------------------
num_train_samples <- nrow(x_train)
x_train_tf <- array_reshape(x_train, c(num_train_samples, lookback, 1))
y_train_scaled_tf <- as.matrix(y_train_scaled)

num_val_samples <- nrow(x_test)
x_test_tf <- array_reshape(x_test, c(num_val_samples, lookback, 1))
y_val_scaled_tf <- as.matrix(y_test_scaled)


## 3.3 Define Optimizer and Loss Functions ---------------------
tf_learning_rate <- 0.00001 # same as R

optimizer_tf <- tf$keras$optimizers$SGD(learning_rate = tf_learning_rate)

# Custom loss: SUM (0.5 * (y_true - y_pred)^2) for gradient calculation
custom_loss_tf_sum_half_mse <- function(y_true, y_pred) {
  squared_difference <- tf$square(y_true - y_pred)
  sum_half_squared_difference <- tf$reduce_sum(0.5 * squared_difference)
  return(sum_half_squared_difference)
}


## 3.4 TensorFlow BGD Training Loop ------------------------------

# -----------------------
# Custom Training Step Function
# -----------------------
train_step_tf <- function(model, optimizer, x_batch, y_batch, loss_for_grads_fn) {
  with(tf$GradientTape() %as% tape, {
    predictions <- model(x_batch, training = TRUE)
    loss_value_for_grads <- loss_for_grads_fn(y_batch, predictions)
  })
  grads <- tape$gradient(loss_value_for_grads, model$trainable_variables)
  optimizer$apply_gradients(zip_lists(grads, model$trainable_variables))
  
  return(list(loss = loss_value_for_grads, grads = grads))  # return both
}

# -----------------------
# Training Setup
# -----------------------
tf_epochs <- 100 # same as R

# For storing losses and gradient sums
tf_train_custom_sum_loss_epoch <- numeric(tf_epochs)
tf_train_reported_mse_epoch    <- numeric(tf_epochs)
tf_val_reported_mse_epoch      <- numeric(tf_epochs)
grad_sums_tf <- numeric(tf_epochs)

# Optional: Plot setup
plot(1, type="n", xlim=c(1, tf_epochs), ylim=c(0, 1), 
     xlab="Epoch", ylab="MSE", main="Training vs Validation MSE")
legend("topright", legend=c("Train MSE", "Val MSE"), col=c("blue", "red"), lty=1)

cat("Starting TensorFlow BGD Training (with manually set R initial weights)...\n")

# -----------------------
# Training Loop
# -----------------------
for (epoch in 1:tf_epochs) {
  # --- 1. Training step
  train_result <- train_step_tf(
    model_tf_manual_weights,
    optimizer_tf,
    x_train_tf,
    y_train_scaled_tf,
    custom_loss_tf_sum_half_mse
  )
  
  current_epoch_sum_loss <- as.numeric(train_result$loss)
  tf_train_custom_sum_loss_epoch[epoch] <- current_epoch_sum_loss
  
  # --- 2. Sum of gradients (for comparison, L2)
  
  # Note: comparison to be found in RMD file
  grads <- train_result$grads
  grad_sums_tf[epoch] <- sum(sapply(grads, function(g) {
    if (is.null(g)) return(0)
    sum(g$numpy()^2)
  }))
  
  # --- 3. Train and validation MSE
  train_preds_tf_epoch <- model_tf_manual_weights(x_train_tf, training = FALSE)
  current_train_mse <- loss_fn_tf_reporting_mean_mse(y_train_scaled_tf, train_preds_tf_epoch)
  tf_train_reported_mse_epoch[epoch] <- as.numeric(current_train_mse)
  
  val_preds_tf_epoch <- model_tf_manual_weights(x_test_tf, training = FALSE)
  current_val_mse <- loss_fn_tf_reporting_mean_mse(y_val_scaled_tf, val_preds_tf_epoch)
  tf_val_reported_mse_epoch[epoch] <- as.numeric(current_val_mse)
  
  # --- 4. Live plot update
  if (epoch %% 5 == 0 || epoch == 1) {
    lines(1:epoch, tf_train_reported_mse_epoch[1:epoch], type="l", col="blue")
    lines(1:epoch, tf_val_reported_mse_epoch[1:epoch], type="l", col="red")
  }
  
  # --- 5. Console output + safety check
  if (epoch %% 10 == 0 || epoch == 1) {
    cat(sprintf("TF Epoch %d: Train = %.6f, Val = %.6f\n",
                epoch, current_train_mse, current_val_mse))
    
    if (is.nan(current_epoch_sum_loss) || is.infinite(current_epoch_sum_loss)) {
      cat("⚠️  EXPLODING GRADIENTS DETECTED IN TENSORFLOW (NaN/Inf loss)!\n")
      cat("💡  Consider adding gradient clipping inside train_step_tf function.\n")
      break
    }
  }
}

# --- End of TensorFlow Section ---

# --- for comparison of gradient and mse please refer to me the RMD file ---


