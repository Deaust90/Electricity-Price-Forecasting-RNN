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
