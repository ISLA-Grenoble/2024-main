
load("ESL.mixture.rda") # this dataset was downloaded from https://hastie.su.domains/ElemStatLearn/

# load the data points used for training
X_train <- ESL.mixture$x
y_train <- ESL.mixture$y
df_train <- data.frame(cbind(y_train, X_train))
colnames(df_train) <- c('y_train', 'X1', 'X2')

# create a data frame for the test dataset
x1_array <- ESL.mixture$px1
x2_array <- ESL.mixture$px2
X_test <- expand.grid(x1_array, x2_array)
df_test <- data.frame(X_test)
colnames(df_test) <- c('X1', 'X2')

# true conditional probability P(Y = 1 | X = x) -- this is what we are aiming for
prob_true <- matrix(ESL.mixture$prob, nrow=length(x1_array), ncol=length(x2_array))

# estimating the logistic regression classifier for this data
logreg <- glm(y_train ~ ., data=df_train, family=binomial)
y_pred <- predict.glm(logreg, newdata=df_test, type="response")
# getting the probability predictions for each point in the feature space
prob_logreg <- matrix(y_pred, nrow=length(x1_array), ncol=length(x2_array))

# estimating the LDA classifier for this data
library("MASS")
ldaclf <- lda(y_train ~ ., data=df_train)
y_pred <- predict(ldaclf, newdata=df_test)
# getting the probability predictions for each point in the feature space
prob_ldaclf <- matrix(y_pred$posterior[,1], nrow=length(x1_array), ncol=length(x2_array))

# estimating the Naive Bayes classifier for this data
library("e1071")
naivclf <- naiveBayes(y_train ~ ., data=df_train)
y_pred <- predict(naivclf, newdata=df_test, type="raw")
# getting the probability predictions for each point in the feature space
prob_naivclf <- matrix(y_pred[,1], nrow=length(x1_array), ncol=length(x2_array))

# estimate the kNN classifier for this data
library("class")
prob_knn_15 <- knn(train=df_train[,c(2,3)], test=df_test, cl=df_train[,1], k=15, prob=TRUE)
# getting the probability predictions for each point in the feature space
prob_knn_15 <- matrix(prob_knn_15, nrow=length(x1_array), ncol=length(x2_array))

method_list <- list(prob_true, prob_logreg, prob_ldaclf, prob_naivclf, prob_knn_15)
labels_list <- list("BAYES", "LOGREG", "LDA", "NAIVEB", "KNN")
titles_list <- list("Bayes decision boundary", "Logistic regression", "Linear Discriminant Analysis", "Naive Bayes classifier", "kNN with k = 15")
  
for (i in 1:5)
{
  filename <- paste("figure1-MixtureExample-", labels_list[i], ".pdf", sep="")
  pdf(filename)
  # plotting the data points for each class
  plot(X, col='white', xlab="X1", ylab="X2", main=titles_list[i])
  points(X[y == 0,], col='#1f77b4', pch=16)
  points(X[y == 1,], col='#ff7f0e', pch=16)
  contour(x1_array, x2_array, method_list[[i]], col='black', levels=0.50, lwd=2.0, lty=2, add=TRUE, drawlabels=FALSE)
  dev.off()
}

# doing KNN with k = 03
prob_knn_03 <- knn(train=df_train[,c(2,3)], test=df_test, cl=df_train[,1], k=3, prob=TRUE)
# getting the probability predictions for each point in the feature space
prob_knn_03 <- matrix(prob_knn_03, nrow=length(x1_array), ncol=length(x2_array))
# doing KNN with k = 75
prob_knn_75 <- knn(train=df_train[,c(2,3)], test=df_test, cl=df_train[,1], k=75, prob=TRUE)
# getting the probability predictions for each point in the feature space
prob_knn_75 <- matrix(prob_knn_75, nrow=length(x1_array), ncol=length(x2_array))

pdf("figure2-CompareKNN.pdf", width=15, height=5)
par(mfrow=c(1, 3))
# plotting the data points for each class
plot(X, col='white', xlab="X1", ylab="X2", main="kNN with k = 03")
points(X[y == 0,], col='#1f77b4', pch=16)
points(X[y == 1,], col='#ff7f0e', pch=16)
contour(x1_array, x2_array, prob_knn_03, col='black', levels=0.50, lwd=2.0, lty=2, add=TRUE, drawlabels=FALSE)
# plotting the data points for each class
plot(X, col='white', xlab="X1", ylab="X2", main="kNN with k = 15")
points(X[y == 0,], col='#1f77b4', pch=16)
points(X[y == 1,], col='#ff7f0e', pch=16)
contour(x1_array, x2_array, prob_knn_15, col='black', levels=0.50, lwd=2.0, lty=2, add=TRUE, drawlabels=FALSE)
# plotting the data points for each class
plot(X, col='white', xlab="X1", ylab="X2", main="kNN with k = 75")
points(X[y == 0,], col='#1f77b4', pch=16)
points(X[y == 1,], col='#ff7f0e', pch=16)
contour(x1_array, x2_array, prob_knn_75, col='black', levels=0.50, lwd=2.0, lty=2, add=TRUE, drawlabels=FALSE)
dev.off()

pdf("figure3-LDA-Ellipse.pdf")
# plotting the data points for each class
library(scales)
ldaclf <- lda(y_train ~ ., data=df_train)
y_pred <- predict(ldaclf, newdata=df_test)
prob_ldaclf <- matrix(y_pred$posterior[,1], nrow=length(x1_array), ncol=length(x2_array))
plot(X, col='white', xlab="X1", ylab="X2", main='LDA')
points(X[y == 0,], col=alpha('#1f77b4', 0.2), pch=16)
points(X[y == 1,], col=alpha('#ff7f0e', 0.2), pch=16)
library("ellipse")
C = 0.5*(cov(X_train[y_train == 1,]) + cov(X_train[y_train == 0,]))
m = ldaclf$means
points(m[1,1], m[1,2], col='#1f77b4', pch=16, cex=1.5)
points(m[2,1], m[2,2], col='#ff7f0e', pch=16, cex=1.5)
lines(ellipse(x=C, centre=m[1,]), lwd=3.0, col='#1f77b4')
lines(ellipse(x=C, centre=m[2,]), lwd=3.0, col='#ff7f0e')
contour(x1_array, x2_array, prob_ldaclf, col='black', levels=0.50, lwd=2.0, lty=2, add=TRUE, drawlabels=FALSE)
dev.off()

