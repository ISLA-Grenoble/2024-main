# set the seed for reproducible results
set.seed(1)
# how many samples to generate
N = 200
# choose the values of xi
X <- seq(-5, 5, length=N)
# generate the observations yi
Y <- X - X^2 + X^3 + rnorm(N, mean=0, sd=20)
# randomly split the dataset into CV and test partitions
idx_random = sample(1:N, replace=FALSE)
N_test = 50
X_test <- X[idx_random[1:N_test]]
Y_test <- Y[idx_random[1:N_test]]
N_cv = N - N_test
X_cv <- X[idx_random[(N_test+1):N]]
Y_cv <- Y[idx_random[(N_test+1):N]]
# how many folds
K = 5
# choose the maximum degree of the polynomial to approximate data
degree_max = 10
# create array to store cvError on each degree
cvError_degree <- vector(mode="numeric",length=K)
# loop through the degrees
for (degree in 1:degree_max)
{
  # create array to store the errors for each fold
  cvError_k <- vector(mode="numeric",length=K)
  for (k in 1:K)
  {
    # choose the indices of the samples for the fold k-th fold
    idx_val <- k + seq(0, N_cv-K, by=K)
    # get the training dataset
    X_train <- X_cv[-idx_val]; 
    Y_train <- Y_cv[-idx_val]
    # get the testing dataset
    X_val <- X_cv[idx_val]
    Y_val <- Y_cv[idx_val]
    # estimate the parameters of a polynomial fit to the data
    x = X_train
    y = Y_train
    f <- lm(y ~ poly(x, degree=degree, raw=TRUE))
    # get the prediction error to estimate data from validation set
    error <- predict(object=f, newdata=data.frame(x=X_val)) - Y_val
    # calculate the mean squared error for this fold
    cvError_k[k] <- mean(error^2)
  }
  # get the average cvError across the folds
  cvError_degree[degree] = mean(cvError_k)
}
# get the degree with the minimum CV error
degree_min = which.min(cvError_degree)
# estimate a model on this degree with whole CV dataset
x = X_cv
y = Y_cv
f <- lm(y ~ poly(x, degree=degree_min, raw=TRUE))
# estimate the error of the model on the test dataset
error <- predict(object=f, newdata=data.frame(x=X_test)) - Y_test
testError = mean(error^2)

