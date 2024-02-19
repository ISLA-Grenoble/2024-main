set.seed(0)
# Number of variables
pt = 201
# Number of predictors
p = pt - 1
# Sample size
n = 30 * p
D = matrix(rnorm(n*pt), nrow=n, ncol=pt)
D = data.frame(D)
names(D)[pt] = "Y"
reg = lm(Y~., data=D)