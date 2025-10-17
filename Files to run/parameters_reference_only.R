# 1. Model Architecture
model_type          <- "SimpleRNN"
input_timesteps     <- 24        # = look_back
input_features      <- 1         # = input_dim (at each timestep)

hidden_units        <- 16        # = hidden_dim
hidden_activation   <- "tanh"
use_bias_hidden     <- TRUE      # (b_h exists)

output_units        <- 1         # = output_dim
output_activation   <- "linear"
use_bias_output     <- TRUE      # (b_y exists)

# 2. Weight Initializers (Target for TensorFlow)
#    (Xavier/Glorot Uniform for W_xh, W_hh, W_hy; Zeros for b_h, b_y)
#    TensorFlow equivalents:
#    - kernel_initializer (for W_xh, W_hy): 'glorot_uniform'
#    - recurrent_initializer (for W_hh): 'glorot_uniform' (Note: TF default for SimpleRNN might be 'orthogonal')
#    - bias_initializer (for b_h, b_y): 'zeros'
#    - Seed for R init: set.seed(1234) # For exact weight matching, transfer R weights to TF

# 3. Data
#    - X_train shape for TF: (num_samples, 24, 1)
#    - y_train shape for TF: (num_samples, 1) or (num_samples,)

# 4. Training
optimizer           <- "SGD"     # (To mimic BGD in TF)
learning_rate       <- 0.00001
loss_function       <- "mean_squared_error" # (mse)
batch_size          <- N_train_samples # (To make SGD act as BGD)
epochs              <- 100