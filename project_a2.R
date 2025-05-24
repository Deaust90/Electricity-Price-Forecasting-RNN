library(data.table)   
library(dplyr)        
library(lubridate)    
library(zoo)
library(tensorflow)
library(keras)
library(tidyverse)

# 1. Data Preprocessing ----------------------------------

# Unzip and Load
csv_files <- list.files(
  path = "C:\\Users\\udwal\\Documents\\Studies_Viadrina\\semester_3\\DeepLearning\\project\\data_rnn",
  pattern = "\\.csv$", 
  full.names = TRUE)

# Combine
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
complete_data$Price <- na.approx(complete_data$Price, na.rm = FALSE)

sum(is.na(complete_data$Price)) # 0 NAs, following redunant but just to keep it same

# Fill leading/trailing NAs (if any remain)
complete_data$Price <- na.locf(complete_data$Price, na.rm = FALSE)
complete_data$Price <- na.locf(complete_data$Price, fromLast = TRUE, na.rm = FALSE)

# Remove duplicate timestamps (keep first occurrence)
complete_data <- complete_data %>% 
  distinct(DateTime, .keep_all = TRUE)

# Verify no missing values
sum(is.na(complete_data$Price))

# 2. Prepare Data for the RNN ----------------------------

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

# making sure scale is correct
par(mfrow = c(2, 1))

# Before scaling
plot(y_train, type = "l", main = "Target over Time (Original)", ylab = "Original Price")

# After scaling
plot(y_train_scaled, type = "l", main = "Target over Time (Standardized)", ylab = "Scaled Price")
abline(h = 0, col = "red", lty = 2)

par(mfrow = c(1, 1))

#   learning_rate_R (e.g., 0.00001)
## 3.1 TensorFlow Model Definition and Manual Weight Setting ----------------

input_dim_R <- 1    
hidden_dim_R <- 16  
output_dim_R <- 1
seed_R <- 1234

set.seed(seed_R)

# 2. Call your rnn.weights.init function:
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

# Define the model structure
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
    name = "output_layer_manual"  # No initializers specified
  )

# --- Note: custom RNN needs to be added to environment --
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

# bias (hidden)  ←-- most common culprit
tf_b_h_from_R <- as.numeric(initial_weights_list_R$b_h)   # drop all dims
if (length(tf_b_h_from_R) == 1)                           # scalar? replicate
  tf_b_h_from_R <- rep(tf_b_h_from_R, tf_hidden_units)    # 16-long vector
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
tf_learning_rate <- 0.00001 # Use the same learning rate

optimizer_tf <- tf$keras$optimizers$SGD(learning_rate = tf_learning_rate)

# Custom loss: SUM (0.5 * (y_true - y_pred)^2) for gradient calculation
custom_loss_tf_sum_half_mse <- function(y_true, y_pred) {
  squared_difference <- tf$square(y_true - y_pred)
  sum_half_squared_difference <- tf$reduce_sum(0.5 * squared_difference)
  return(sum_half_squared_difference)
}

# Standard MSE for reporting
loss_fn_tf_reporting_mean_mse <- tf$keras$losses$MeanSquaredError()


## 3.4 Custom Training Step Function --------------------------------

train_step_tf <- function(model, optimizer, x_batch, y_batch, loss_for_grads_fn) {
  with(tf$GradientTape() %as% tape, {
    predictions <- model(x_batch, training = TRUE)
    loss_value_for_grads <- loss_for_grads_fn(y_batch, predictions)
  })
  grads <- tape$gradient(loss_value_for_grads, model$trainable_variables)
  
  optimizer$apply_gradients(zip_lists(grads, model$trainable_variables))
  return(loss_value_for_grads)
}


## 3.5 TensorFlow BGD Training Loop ------------------------------
tf_epochs <- 100 # Use the same number of epochs as R model

# For storing losses and initial gradients
tf_train_custom_sum_loss_epoch <- numeric(tf_epochs)
tf_train_reported_mse_epoch    <- numeric(tf_epochs)
tf_val_reported_mse_epoch      <- numeric(tf_epochs)
initial_summed_grads_tf_list   <- NULL

print("Starting TensorFlow BGD Training (with manually set R initial weights)...")
for (epoch in 1:tf_epochs) {
  # Capture initial gradients in the first epoch (before any weight updates)
  if (epoch == 1) {
    with(tf$GradientTape() %as% tape_init_grad, {
      predictions_init <- model_tf_manual_weights(x_train_tf, training = FALSE)
      loss_init <- custom_loss_tf_sum_half_mse(y_train_scaled_tf, predictions_init)
    })
    initial_summed_grads_tf_list <- tape_init_grad$gradient(
      loss_init, model_tf_manual_weights$trainable_variables
    )
    # This initial_summed_grads_tf_list is key for comparing with R's initial gradients.
  }
  
  # Perform one BGD step (process all training data)
  current_epoch_sum_loss <- train_step_tf(
    model_tf_manual_weights,
    optimizer_tf,
    x_train_tf,
    y_train_scaled_tf,
    custom_loss_tf_sum_half_mse
  )
  tf_train_custom_sum_loss_epoch[epoch] <- as.numeric(current_epoch_sum_loss)
  
  # Calculate reported MSE for training set for this epoch
  train_preds_tf_epoch <- model_tf_manual_weights(x_train_tf, training = FALSE)
  current_train_mse <- loss_fn_tf_reporting_mean_mse(y_train_scaled_tf, train_preds_tf_epoch)
  tf_train_reported_mse_epoch[epoch] <- as.numeric(current_train_mse)
  
  # Calculate reported MSE for validation set for this epoch
  val_preds_tf_epoch <- model_tf_manual_weights(x_test_tf, training = FALSE)
  current_val_mse <- loss_fn_tf_reporting_mean_mse(y_val_scaled_tf, val_preds_tf_epoch)
  tf_val_reported_mse_epoch[epoch] <- as.numeric(current_val_mse)
  
  # Print progress
  if (epoch %% 10 == 0 || epoch == 1) {
    cat(sprintf("TF Epoch %d: Train = %.6f, Val = %.6f\n",
                epoch, 
                as.numeric(current_train_mse), 
                as.numeric(current_val_mse)))
    
    if (is.nan(as.numeric(current_epoch_sum_loss)) || is.infinite(as.numeric(current_epoch_sum_loss))) {
      print("EXPLODING GRADIENTS DETECTED IN TENSORFLOW (NaN/Inf loss)!")
      print("Consider adding gradient clipping inside train_step_tf function.")
      break # Stop training if it explodes
    }
  }
}
print("TensorFlow BGD Training Finished.")

## Plotting Train and Validation MSE -------------------------

# First, check if the lengths match the number of epochs
if (length(tf_train_reported_mse_epoch) != tf_epochs || length(tf_val_reported_mse_epoch) != tf_epochs) {
  warning("Mismatch in length of loss arrays and tf_epochs. Plot might be incorrect.
           Using the minimum length found for plotting.")
  epochs_to_plot <- min(tf_epochs, length(tf_train_reported_mse_epoch), length(tf_val_reported_mse_epoch))
} else {
  epochs_to_plot <- tf_epochs
}

# Ensure we only use the valid portion of the arrays if there was a mismatch or early stop
train_losses_to_plot <- tf_train_reported_mse_epoch[1:epochs_to_plot]
val_losses_to_plot <- tf_val_reported_mse_epoch[1:epochs_to_plot]
epoch_numbers <- 1:epochs_to_plot

plot_data <- data.frame(
  Epoch = rep(epoch_numbers, 2),
  MSE = c(train_losses_to_plot, val_losses_to_plot),
  Dataset = factor(rep(c("Training (Scaled)", "Validation (Scaled)"), each = epochs_to_plot))
)

# Create the Convergence Plot using ggplot2
convergence_plot_scaled <- ggplot(plot_data, aes(x = Epoch, y = MSE, color = Dataset, linetype = Dataset)) +
  geom_line(linewidth = 1) +  
  labs(
    title = "TensorFlow Model Convergence (Scaled Data)",
    x = "Epoch",
    y = "Mean Squared Error (on Scaled Data)",
    color = "Dataset",        
    linetype = "Dataset"   
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"), 
    legend.position = "top", # Position legend at the top
    legend.title = element_text(face = "bold")
  )

print(convergence_plot_scaled)


# --- End of TensorFlow Section ---




