# Home Assignment in Deep Learning and Neural Networks

This repository contains the final assignment for the *Deep Learning and Neural Networks* course. The project explores Recurrent Neural Networks (RNNs) implemented both in base R and using TensorFlow, along with a numerical gradient check for model verification.

## Authors
- Udval Oyunsaikhan  
- Cadena Bolaños Andres Felipe

## Repository Structure

- `parameters_reference_only.R`  
  ➤ Contains a reference list of hyperparameters used throughout the project.

- `project_a1.R`  
  ➤ Implementation of an RNN using base R.

- `project_a2.R`  
  ➤ Implementation of an RNN using TensorFlow (via `tensorflow` package in R).

- `gradient_check.R`  
  ➤ Performs a numerical gradient check to validate the gradient computations.

- `funcs_mlr.R`  
➤ Contains a reference list of hyperparameters used throughout the project. Please make sure to use this file as custom BGD function has been revised from the course provided file.

## Instructions to Run

To fully replicate the environment and results:

1. **Load all files into your R environment.**
2. Check out the HTML for a nice glance of all submission.
3. **Run `project_a1.R` first** — this initializes and trains the base R implementation of the RNN.
4. **Then run `project_a2.R`** — to run the RNN with TensorFlow.
5. You may consult:
   - `parameters_reference_only.R` for parameter tuning or documentation.
   - `gradient_check.R` to manually inspect the correctness of gradient calculations.

## Requirements

- R 4.x or higher
- [tensorflow](https://tensorflow.rstudio.com/) package installed in R
- Basic understanding of RNNs and backpropagation through time

---
