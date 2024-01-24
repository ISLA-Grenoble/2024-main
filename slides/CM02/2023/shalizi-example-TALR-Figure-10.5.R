sim.non.slr <- function(n, do.test = FALSE) {
  x <- rexp(n, rate = 0.5)
  y <- (x - 1)^2 * runif(n, min = 0.8, max = 1.2)
  if (!do.test) {
    return(data.frame(x = x, y = y))
  } else {
    # Fit a linear model, run F test, return p-value
    return(anova(lm(y ~ x))[["Pr(>F)"]][1])
  }
}

not.slr <- sim.non.slr(n = 200)
plot(y ~ x, data = not.slr)
curve((x - 1)^2, col = "blue", add = TRUE)
abline(lm(y ~ x, data = not.slr), col = "red")