## Norms function
norms.calc <- function(beta){

    beta_l1 <- sum(abs(beta))
    beta_l2 <- sqrt(sum(abs(beta)^2))
    
    return(c(beta_l1, beta_l2))
}


## MSE function
mse.calc <- function(y_est, y_true){
  MSE <- mean((y_est - y_true)^2)
  return(MSE)
}

## TE function
te.calc <- function(y_est, y_true){
    err <- scale(y_est - y_true, scale = FALSE)
    te <- sqrt(sum(t(err)%*%err)/(length(y_true)-1))
    return(te)
}

## Standardization function
standardize <- function(value){
  value_stand <- sqrt(sum((value-mean(value))^2)/length(value))

  return(value_stand)
}

## LOOCV
loocv.glmnet <- function(X, y, alpha = 0, lambda, ...){
  # number of observations (training data)
  n <- nrow(X)
  # number of lambda elements
  l <- length(lambda)
  # empty matrix for the squared errors
  SE_val <- matrix(, n, l)
  # for each validation fold (LOOCV has n validation folds)
  for(i in 1:n){
    # fit the model on the rest of the data
    mod_fit <- glmnet(X[-i,], y[-i], alpha = alpha, lambda = lambda)
    # calculate the squared error for the validation data 
    SE_val[i,] <- as.numeric(y[i] - (X[i,]%*%as.matrix(mod_fit$beta)))^2
  }
  # calculate the mean over all validation folds
  MSE_val <- colMeans(SE_val)
  # find the optimal lambda (min MSE)
  lambda_min <- lambda[which.min(MSE_val)]
  
  return(lambda_min)
}

## IC function
ic.calc <- function(kappa, RSS, DF, n){
  
  IC_value <- log(RSS) + kappa*DF/n
  return(IC_value)
  
}

## Standardize X function
X.standardize <- function(X_train, X_test){
  
  # calculate the values for the mean and standard deviation
  X_mean <- apply(X_train, 2, mean)
  X_sd <- apply(X_train, 2, sd)
  
  # standardize the data
  X_train_std <- scale(X_train, center = X_mean, scale = X_sd)
  X_test_std <- scale(X_test, center = X_mean, scale = X_sd)
  
  return(list(X_train_std, X_test_std))
}

## Weights initialization function
weights.init <- function(input_nodes, hidden_nodes, output_nodes){
  
  weights_init <- runif(input_nodes * hidden_nodes + 
                          hidden_nodes * output_nodes, -1, 1)
  
  return(weights_init)
}

## Neural Network function
neural_net <- function(weights, X, hidden_nodes, output_nodes) {

  act1 <- function(x)
    (1 / (1 + exp(-x)))
  act2 <- function(x)
    (1 * x)
  
  w <-
    matrix(weights[1:(ncol(X) * hidden_nodes)], ncol(X), hidden_nodes)
  v <-
    matrix(weights[-(1:(ncol(X) * hidden_nodes))], hidden_nodes, output_nodes)
  
  hid <- X %*% w
  hid_act <- act1(hid)
  
  out <- hid_act %*% v
  out_act <- act2(out)
  
  return(list(
    hid = hid,
    hid_act = hid_act,
    out = out,
    y_est = out_act,
    w = w,
    v = v
  ))
}

## Cost function 
cost_ss <- function(weights, y, X, nn, hid_n, out_n, ...) {
  nn_res <- nn(weights, X, hid_n, out_n)
  return(sum(0.5 * (y - nn_res$y_est) ^ 2))
}


## Cost derivatives function
costNN_deriv <- function(weights, y, X, nn, hid_n, out_n, ...) {
  nn_res <- nn(weights, X, hid_n, out_n)
  
  act1_d <- function(x) {
    exp(-x) / ((1 + exp(-x)) ^ 2)
  }
  act2_d <- function(x) {
    1
  }
  
  delta2 <- -(y - nn_res$y_est) * act2_d(nn_res$out)
  v_deriv <- t(nn_res$hid_act) %*% delta2
  
  delta1 <- (delta2 %*% t(nn_res$v)) * act1_d(nn_res$hid)
  w_deriv <- t(X) %*% delta1
  
  return(c(w_deriv, v_deriv))
  
}


## (Full) Batch Gradient Descent 
bgd <- function(weights, y, X, hid_n, out_n, nn, costderiv, 
                epochs = 3000, lr = 0.00001, ...){
  
  for(i in 1:epochs){
    
    deriv <- costderiv(weights, y, X, nn, hid_n, out_n, ...)
    
    weights <- weights - lr*deriv
    
  }
  
  return(weights)
  
}

## Weights with Bias
weights.init.bias <- function(input_nodes, hidden_nodes, output_nodes){
  
  weights_init_bias <-  runif(input_nodes*hidden_nodes 
                              + (hidden_nodes + 1)*output_nodes, -1, 1)
  
  return(weights_init_bias)
}

## Neural Network with Bias
neural_net_bias <- function(weights, X, hidden_nodes, output_nodes) {

  act1 <- function(x)
    (1 / (1 + exp(-x)))
  act2 <- function(x)
    (1 * x)
  
  w <-
    matrix(weights[1:(ncol(X) * hidden_nodes)], ncol(X), hidden_nodes)
  v <-
    matrix(weights[-(1:(ncol(X) * hidden_nodes))], hidden_nodes + 1, output_nodes)
  
  hid <- X %*% w
  hid_act <- cbind(rep(1, nrow(X)), act1(hid))
  
  out <- hid_act %*% v
  out_act <- act2(out)
  
  return(list(
    hid = hid,
    hid_act = hid_act,
    out = out,
    y_est = out_act,
    w = w,
    v = v
  ))
}

## Cost Function with Regularization
cost_reg <- function(weights, y, X, nn, hid_n, out_n, lambda, ...) {
  nn_res <- nn(weights, X, hid_n, out_n)
  return(sum(0.5 * (y - nn_res$y_est) ^ 2) + (lambda / 2) * (sum(nn_res$w ^ 2) + sum(nn_res$v ^ 2)))
}

## Derivation Cost Function Neural Network with Regularization
costNN_deriv_reg <-
  function(weights, y, X, nn, hid_n, out_n, lambda, ...) {
    nn_res <- nn(weights, X, hid_n, out_n)
    
    act1_d <- function(x) {
      exp(-x) / ((1 + exp(-x)) ^ 2)
    }
    act2_d <- function(x) {
      1
    }
    
    delta2 <- -(y - nn_res$y_est) * act2_d(nn_res$out)
    v_deriv <- t(nn_res$hid_act) %*% delta2 + lambda * nn_res$v
    
    v <- nn_res$v[-1, ]
    
    delta1 <- (delta2 %*% t(v)) * act1_d(nn_res$hid)
    w_deriv <- t(X) %*% delta1 + lambda * nn_res$w
    
    return(c(w_deriv, v_deriv))
    
  }

## Newton's method
newton <- function(weights, y, X, hid_n, out_n, nn, cost, 
                   costderiv, epochs = 30, ...){
  
  for(i in 1:epochs){
    
    deriv <- costderiv(weights, y, X, nn, hid_n, out_n, ...)
    
    deriv2nd <- numDeriv::hessian(func = cost, x = weights, y = y, 
                                  X = X, nn = nn, hid_n = hid_n, out_n = out_n, ...)
    
    lr <-  c((t(deriv)%*%deriv)/(t(deriv)%*%deriv2nd%*%deriv))
    
    weights <- weights - lr*deriv
    
  }
  
  return(weights)
  
}

## Stochastic Gradient Descent (Mini Batch)
sgd <- function(weights, X, y, hid_n, out_n, nn, costderiv,
                epochs = 10000, batch_size = 10, lr = 0.00001, ...){
  
  for(i in 1:epochs){
    
    index_shuffle <- sample(1:nrow(X), nrow(X), replace = FALSE)
    index_groups <- cut(1:nrow(X), 
                        seq(1, nrow(X), batch_size) - 1, labels = FALSE)
    
    for(j in 1:max(index_groups,na.rm=TRUE)){
      
      index_batch <- index_shuffle[which(index_groups==j)]
      
      deriv <- costderiv(weights, y[index_batch], X[index_batch,], nn, 
                         hid_n, out_n, ...)
      
      weights <- weights - lr*deriv
      
    }
    
  }
  
  return(weights)
  
}
