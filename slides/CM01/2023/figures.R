data = read.csv('hospital-infection.txt', header=TRUE, sep='\t')
data_sub = data[(data$Region == 1) | (data$Region == 2),]
data_sub = data_sub[data_sub$Stay < 16,]

pdf(file = "./Figure-01.pdf",   # The directory you want to save the file in
    width = 7, # The width of the plot in inches
    height = 6) # The height of the plot in inches
plot(data_sub$Stay, data_sub$InfctRsk, 
     xlab='Stay', ylab='Infection Risk', pch=16)
dev.off()

summary(lm(InfctRsk~Stay+Age+Xray, data=data_sub))

library("palmerpenguins")
pdf(file = "./Figure-03.pdf",   # The directory you want to save the file in
    width = 8, # The width of the plot in inches
    height = 6) # The height of the plot in inches

plot(penguins$bill_length_mm,
     penguins$bill_depth_mm, 
     col='white',
     xlab='bill length (mm)',
     ylab='bill depth (mm)')

penguins_Adelie = penguins[penguins$species == "Adelie",]
points(penguins_Adelie$bill_length_mm, 
       penguins_Adelie$bill_depth_mm, 
       col='darkorange', pch=16, cex=1.2)

penguins_Gentoo = penguins[penguins$species == "Gentoo",]
points(penguins_Gentoo$bill_length_mm,
       penguins_Gentoo$bill_depth_mm, 
       col='cyan4', pch=16, cex=1.2)

penguins_Chinstrap = penguins[penguins$species == "Chinstrap",]
points(penguins_Chinstrap$bill_length_mm,
       penguins_Chinstrap$bill_depth_mm, 
       col='darkorchid', pch=16, cex=1.2)

dev.off()

pdf(file = "./Figure-07.pdf",   # The directory you want to save the file in
    width = 7, # The width of the plot in inches
    height = 6) # The height of the plot in inches
data = read.csv('hospital-infection.txt', header=TRUE, sep='\t')
data_sub = data[(data$Region == 1) | (data$Region == 2),]
data_sub = data_sub[data_sub$Stay < 16,]
plot(data_sub$Stay, data_sub$InfctRsk, 
     xlab='Stay', ylab='Infection Risk', pch=16)
abline(lm(InfctRsk ~ Stay, data=data))
dev.off()

out = summary(lm(InfctRsk ~ Stay, data=data))
beta_0_hat = out$coefficients[1,1]
beta_0_hat_std = out$coefficients[1,2]
beta_1_hat = out$coefficients[2,1]
beta_1_hat_std = out$coefficients[2,2]

x0 = seq(from=beta_0_hat-5*beta_0_hat_std, 
         to=beta_0_hat+5*beta_0_hat_std, 
         length.out=100)
y0 = dnorm(x0, mean=beta_0_hat, sd=beta_0_hat_std)
x1 = seq(from=beta_1_hat-5*beta_1_hat_std, 
         to=beta_1_hat+5*beta_1_hat_std, 
         length.out=100)
y1 = dnorm(x1, mean=beta_1_hat, sd=beta_1_hat_std)

pdf(file = "./Figure-08.pdf",   # The directory you want to save the file in
    width = 12, # The width of the plot in inches
    height = 5) # The height of the plot in inches
par(mfrow=c(1,2))
plot(x0, y0, col='blue', type='l', lty=1, xlab='beta_0', ylab='')
plot(x1, y1, col='blue', type='l', lty=1, xlab='beta_1', ylab='')
dev.off()

pdf(file = "./Figure-09.pdf",   # The directory you want to save the file in
    width = 12, # The width of the plot in inches
    height = 5) # The height of the plot in inches
res = out$residuals
par(mfrow=c(1,2))
plot(data$Stay, res, xlab='Stay', ylab='residuals')
abline(h = 0, col='grey')
qqnorm(res)
qqline(res)
dev.off()


