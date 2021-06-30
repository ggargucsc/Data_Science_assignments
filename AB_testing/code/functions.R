library(nortest)

plot_metrics <- function(y){
  # plots showing the districution 
  par(mfrow=c(1,2))
  mean.y <- mean(y)
  sd.y <- sd(y)
  hist(y, probability=T)
  y.range <- seq(floor(min(y)), ceiling(max(y)), 0.05)
  x <- dnorm(y.range, mean=mean.y, sd=sd.y)
  lines(y.range, x, col="red")
  qqnorm(y, cex=0.5); qqline(y, col = 2)
  
}

# only in between designs
check_residuals_normality <- function(y, factor, data){
  # check if residuals follows normal distribution
  m = aov(data[[y]]~data[[factor]])
  hist(residuals(m))
  qqnorm(residuals(m))
  qqline(residuals(m))
  print(ad.test(residuals(m)))
}

