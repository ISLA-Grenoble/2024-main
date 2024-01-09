graphics.off()

set.seed(42)

# fix the number of samples to consider for each xi
Ni = 100
# choose at random the values for the xi
x = seq(1, 5, length.out=101)

# create empty data frame to populate in the for loop
df = data.frame()
for (i in 1:Ni)
{
  # generate noise
  ei = (3*x + 1) * 0.25 * rnorm(100)
  # get the observations from our model corrupted by noise
  yi = (x-1)^2 + ei
  # create dataframe with the results of the ith realization
  dfi = data.frame(x, yi)
  # concatenate the results with previous df
  df = rbind(df, dfi)
}

xcond = 3
ycond = df["yi"][df["x"] == xcond]

econd = c()
for (xi in x)
{
  ycondi = df["yi"][df["x"] == xi]
  econd = rbind(econd, mean(ycondi))
}

# plot the data
plot(df, col="#e8e9ed", type="p", pch=1, cex=.5, xlab="x", ylab="y")
points(rep(xcond, Ni), ycond, col="red", type="p", pch=16, cex=.5, xlab="x", ylab="y")
points(xcond, mean(ycond), col="blue", pch=16, cex=1.5)
lines(x, -6.730 + 4.025*x, col = "black")
lines(x, econd, col = "blue")

