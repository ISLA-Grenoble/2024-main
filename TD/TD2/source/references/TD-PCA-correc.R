## ----setup, include=FALSE, message=FALSE----------------------------
#see full list >knitr::opts_chunk$get()
knitr::opts_chunk$set(echo = TRUE, fig.align="center", prompt = TRUE, comment="")
exercise <- 1


## ---- echo=FALSE----------------------------------------------------
exercise <- exercise+1


## -------------------------------------------------------------------
ecocr <- scale(eco91)
ecocr.acp <- prcomp(ecocr)
# valeurs propres
lambda <- ecocr.acp$sdev^2
matlambda <- rbind(lambda, cumsum(lambda)/sum(lambda))
rownames(matlambda) <- c("lambda", "cumul. proportion")
colnames(matlambda) <- 1:length(lambda)


## ---- echo=FALSE----------------------------------------------------
exercise <- exercise+1


## ---- echo=FALSE----------------------------------------------------
## Load the "cars04" dataset
cars04 <- read.csv("data/cars-fixed04.csv")[,8:18]


## -------------------------------------------------------------------
cars04.pca <- prcomp(cars04, scale=TRUE)
summary(cars04.pca)


## -------------------------------------------------------------------
cars04.pca$rotation[,1:2]


## ---- echo=FALSE, eval=FALSE----------------------------------------
## lambda = cars04.pca$sdev^2
## p = dim(cars04)[2]
## 
## x = (-100:100)/100
## y = sqrt(1-x^2)
## P = cars04.pca$rotation
## for(i in 1:p) P[,i] = P[,i] * sqrt(lambda[i])
## 
## plot(P[,1],P[,2], xlab="Axis 1", ylab = "Axis 2", xlim=c(-1,1), ylim=c(-1,1))
## abline(h=0, v=0)
## lines(x, y)
## lines(x, -y)
## text(P[, 1], P[, 2]-0.05, names(cars04))


## ---- echo=FALSE, fig.width=5, fig.height=5-------------------------
x <- (-100:100)/100
y <- sqrt(1-x^2)
lambda = cars04.pca$sdev^2
P = cars04.pca$rotation
for(i in 1:NCOL(P)) 
  P[,i] <- P[,i] * sqrt(lambda[i])

plot(P[,1], P[,2], xlab="Axis 1", ylab = "Axis 2", xlim=c(-1,1), ylim=c(-1,1))
abline(h=0, v=0)
lines(x, y)
lines(x, -y)
text(P[, 1]+0.1*rep(c(1,-1), length=NCOL(cars04)), 
     P[, 2]-0.1*rep(c(1,-1), length=NCOL(cars04)), names(cars04))



## ---- echo=FALSE, fig.width=5, fig.height=5-------------------------
somerows <- c("Audi RS 6", "Ford Expedition 4.6 XLT", "Nissan Sentra 1.8")
samplrows <- sample(setdiff(rownames(cars04), somerows), 25)
plot(cars04.pca$x[, 1], cars04.pca$x[, 2], xlab="PC1", ylab="PC2", type="n", xlim=c(-8,8), ylim=c(-4,4))
abline(h=0, v=0)
text(cars04.pca$x[samplrows, 1], cars04.pca$x[samplrows, 2], samplrows, cex=0.6, col="grey40")
text(cars04.pca$x[somerows, 1], cars04.pca$x[somerows, 2], somerows, cex=1.05)


## ---- echo=FALSE----------------------------------------------------
exercise <- exercise+1


## ---- echo=FALSE----------------------------------------------------
exercise <- exercise+1


## ---- echo=FALSE----------------------------------------------------
exercise <- exercise+1

