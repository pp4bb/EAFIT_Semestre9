#=======================================================
# Examples of simulation of ARIMA Models
# Time series
# Elaborated by: Manuel Correa Giraldo
#=======================================================

#-------------------------------------------------------
# Packages
#-------------------------------------------------------
install.packages('forecast', dependencies = TRUE)

#-------------------------------------------------------
# Libraries
#-------------------------------------------------------
library(tseries)
library(forecast)
library(plotly)
library(dplyr)


#===========================================================
# White noise
#===========================================================

# Many white noises
k = 15  # Number of time series
n = 200 # Time for each time series

white_noises <- matrix(rnorm(k*n, mean = 0, sd = 1), nrow=k, ncol=n)
matplot(t(white_noises), type = "l", xlab="Time", ylab="Gaussian white noises")

#Plot just one 
plot(white_noises[1,], type = "l", xlab="Time", ylab="Gaussian white noise")

#------------------------------------------------------------
# Unit root tests 
#------------------------------------------------------------
# Augmented Dickey-Fuller test
adf.test(white_noises[1,])

# Phillips-Perron test
pp.test(white_noises[1,])

# KPSS test
kpss.test(white_noises[1,])

#------------------------------------------------------------
# Box-Cox transformation
#------------------------------------------------------------
# Note: Most of the forecasting models in the forecast package 
# automatically transform the series before applying the model, 
# and then re-transform the forecast output back to the original 
# scale.
#-------------------------------------------------------------

BoxCox.lambda(white_noises[1,])
BoxCox.lambda(white_noises[2,])
BoxCox.lambda(white_noises[3,])
BoxCox.lambda(white_noises[4,])
BoxCox.lambda(white_noises[5,])
BoxCox.lambda(white_noises[6,])
BoxCox.lambda(white_noises[7,])
BoxCox.lambda(white_noises[8,])
BoxCox.lambda(white_noises[9,])
BoxCox.lambda(white_noises[10,])


#------------------------------------------------------------
# ACF : White noise
#------------------------------------------------------------
par(mfrow=c(2,2))
acf(white_noises[1,])
acf(white_noises[2,])
acf(white_noises[3,])
acf(white_noises[4,])


#-------------------------------------------------------------
# Histogram : white noise
#-------------------------------------------------------------
par(mfrow=c(2,2))

hist(white_noises[1,], prob = TRUE, main = "Histogram with density curve 1")
lines(density(white_noises[1,]), col = 4, lwd = 2)

hist(white_noises[2,], prob = TRUE, main = "Histogram with density curve 2")
lines(density(white_noises[2,]), col = 4, lwd = 2)

hist(white_noises[3,], prob = TRUE, main = "Histogram with density curve 3")
lines(density(white_noises[3,]), col = 4, lwd = 2)

hist(white_noises[4,], prob = TRUE, main = "Histogram with density curve 4")
lines(density(white_noises[4,]), col = 4, lwd = 2)

#===========================================================
# Simulated random walk
#===========================================================
# Many random walks

k = 15  # Number of time series
n = 200 # Time for each time serie

epsilon <- matrix(rnorm(k*n, mean = 0, sd = 1), nrow=k, ncol=n)
random_walks <- matrix(, nrow=k, ncol=n)

for(i in 1:k){
  random_walks[i,1] <- 0
}


for(i in 1:k){
  for(t in 2:n){ 
    random_walks[i,t] <- random_walks[i,t-1] + epsilon[i,t]
    }
}

par(mfrow=c(1,1))
matplot(t(random_walks), type = "l", xlab="Time", ylab="Random walks")

#------------------------------------------------------------
# Unit root tests 
#------------------------------------------------------------
# Augmented Dickey-Fuller test
adf.test(random_walks[1,])
adf.test(random_walks[10,])

# Phillips-Perron test
pp.test(random_walks[1,])
pp.test(random_walks[10,])

# KPSS test
kpss.test(random_walks[1,])
kpss.test(random_walks[10,])

#------------------------------------------------------------
# Box-Cox transformation
#------------------------------------------------------------
BoxCox.lambda(random_walks[1,])
BoxCox.lambda(random_walks[2,])
BoxCox.lambda(random_walks[3,])
BoxCox.lambda(random_walks[4,])
BoxCox.lambda(random_walks[5,])
BoxCox.lambda(random_walks[6,])
BoxCox.lambda(random_walks[7,])
BoxCox.lambda(random_walks[8,])
BoxCox.lambda(random_walks[9,])
BoxCox.lambda(random_walks[10,])

#------------------------------------------------------------
# ACF : random walks
#------------------------------------------------------------
par(mfrow=c(2,2))
acf(random_walks[2,])
acf(random_walks[4,])
acf(random_walks[8,])
acf(random_walks[10,])

#------------------------------------------------------------
# PACF : random walks
#------------------------------------------------------------
par(mfrow=c(2,2))
pacf(random_walks[2,])
pacf(random_walks[4,])
pacf(random_walks[8,])
pacf(random_walks[10,])

#----------------------------------------------------------
# Estimation of model: AR(1) for Random walk
#-----------------------------------------------------------
rwalk_ar <- ar(random_walks[11,])
rwalk_ar

#---------------------------------------------------------
# Check residuals for estimation of random walk
#----------------------------------------------------------
checkresiduals(rwalk_ar)

#----------------------------------------------------------
# Diff random walk
#----------------------------------------------------------
diff_rwalks <- matrix(, nrow=k, ncol=n-1)

for(i in 1:k){
 diff_rwalks[i,] <- diff(random_walks[i,]) 
  }

par(mfrow=c(1,1))
matplot(t(diff_rwalks), type = "l", xlab="Time", ylab="Diff random walks")

#------------------------------------------------------------
# Unit root tests 
#------------------------------------------------------------
# Augmented Dickey-Fuller test
adf.test(diff_rwalks[1,])
adf.test(diff_rwalks[10,])

# Phillips-Perron test
pp.test(diff_rwalks[1,])
pp.test(diff_rwalks[10,])

# KPSS test
kpss.test(diff_rwalks[1,])
kpss.test(diff_rwalks[10,])

#------------------------------------------------------------
# ACF : random walks
#------------------------------------------------------------
par(mfrow=c(2,2))
acf(diff_rwalks[2,])
acf(diff_rwalks[4,])
acf(diff_rwalks[8,])
acf(diff_rwalks[10,])

#------------------------------------------------------------
# PACF : random walks
#------------------------------------------------------------
par(mfrow=c(2,2))
pacf(diff_rwalks[2,])
pacf(diff_rwalks[4,])
pacf(diff_rwalks[8,])
pacf(diff_rwalks[10,])


#============================================================
# Autoregressive models
#=============================================================

#-------------------------------------------------------------
# Simulated data for AR(1) model with same parameter
#------------------------------------------------------------
set.seed(1234)

k <- 10  # Number of time series
n <- 300 # Time for each time serie
phi.1 <- 0.7 

epsilon <- matrix(rnorm(k*n, mean = 0, sd = 1), nrow=k, ncol=n)
y_ar1 <- matrix(, nrow=k, ncol=n)

for(i in 1:k){
  y_ar1[i,1] <- 0
}

for(i in 1:k){
  for(t in 2:n){ 
    y_ar1[i,t] <- phi.1*y_ar1[i,t-1] + epsilon[i,t]
  }
}

par(mfrow=c(1,1))
matplot(t(y_ar1), type = "l", xlab="Time", ylab="y(t) = 0.7 y(t-1) + e(t)")

#---------------------------------------------------------
# Simulation of AR(1) model with different parameters
#----------------------------------------------------------
phi <- c(0.2, -0.2, 0.5, -0.5, 0.8, -0.8)
epsilon_phi <- epsilon[1:length(phi), ]
y_ar <- matrix(, nrow=length(phi), ncol=n)

for(i in 1:length(phi)){
  y_ar[i,1] <- 0
}

for(i in 1:length(phi)){
  for(t in 2:n){ 
    y_ar[i,t] <-  phi[i]*y_ar[i,t-1] + epsilon[i,t]
  }
}

par(mfrow=c(2,3))
plot(y_ar[1,], type = "l", xlab="Time", ylab="y(t) = 0.2 y(t-1) + e(t)")
plot(y_ar[3,], type = "l", xlab="Time", ylab="y(t) = 0.5 y(t-1) + e(t)")
plot(y_ar[5,], type = "l", xlab="Time", ylab="y(t) = 0.8 y(t-1) + e(t)")
plot(y_ar[2,], type = "l", xlab="Time", ylab="y(t) = -0.2 y(t-1) + e(t)")
plot(y_ar[4,], type = "l", xlab="Time", ylab="y(t) = -0.5 y(t-1) + e(t)")
plot(y_ar[6,], type = "l", xlab="Time", ylab="y(t) = -0.8 y(t-1) + e(t)")

#------------------------------------------------------------
# ACF : AR(1) process with different parameter
#------------------------------------------------------------
par(mfrow=c(2,3))
acf(y_ar[1,])
acf(y_ar[3,])
acf(y_ar[5,])
acf(y_ar[2,])
acf(y_ar[4,])
acf(y_ar[6,])

#------------------------------------------------------------
# PACF : AR(1) process with different parameter
#------------------------------------------------------------
par(mfrow=c(2,3))
pacf(y_ar[1,])
pacf(y_ar[3,])
pacf(y_ar[5,])
pacf(y_ar[2,])
pacf(y_ar[4,])
pacf(y_ar[6,])

#--------------------------------------------------------
# Mean of AR(1) model without constant
#--------------------------------------------------------
mean.ar <- NULL
for(i in 1:length(phi)){mean.ar[i] <- mean(y_ar[i,])}
mean.ar

#--------------------------------------------------------
# Variance of AR(1):sigma^2 (1/(1-phi^2)) 
#--------------------------------------------------------
# Sample variance
var_s.ar <- NULL
for(i in 1:length(phi)){var_s.ar[i] <- var(y_ar[i,])}
var_s.ar

# Theorical variance
var_p.ar <- NULL
for(i in 1:length(phi)){var_p.ar[i] <- (1/(1 - phi[i]^2))}
var_p.ar

#-----------------------------------------------------------
# Simulation of ACF: AR(1) model
#------------------------------------------------------------
phi <- c(0.2, -0.2, 0.5, -0.5, 0.8, -0.8)
k_phi <- 10 # number of lags

rho_ar1 <- matrix(, nrow=length(phi), ncol=k_phi)


for(i in 1:length(phi)){
  for(j in 1:k_phi){ 
    rho_ar1[i,j] <-  phi[i]^(j-1)
  }
}

rho_ar1

par(mfrow=c(2,3))
plot(rho_ar1[1,], type = "o", xlab="Lag", ylab="ACF - phi = 0.2")
plot(rho_ar1[3,], type = "o", xlab="Lag", ylab="ACF - phi = 0.5")
plot(rho_ar1[5,], type = "o", xlab="Lag", ylab="ACF - phi = 0.8")
plot(rho_ar1[2,], type = "o", xlab="Lag", ylab="ACF - phi = -0.2")
plot(rho_ar1[4,], type = "o", xlab="lag", ylab="ACF - phi = -0.5")
plot(rho_ar1[6,], type = "o", xlab="Lag", ylab="ACF - phi = -0.8")

#------------------------------------------------------------
# Simulated data for AR(2) model with parameters 1 and -1/4
#-----------------------------------------------------------

# Check roots of the characteristic polynomial. 
polyroot(c(1, -1, 1/4))


# Simulation using "arima.sim" function in R. 
set.seed(12345)
y_ar2 <- arima.sim(model = list(order = c(2,0,0), ar = c(1, -0.25)), n = 500)

par(mfrow=c(1,1))
plot(y_ar2, ylab = "y(t) = y(t-1) - 0.25 y(t-2) + e(t)")

#-------------------------------------------------------------
# Estimation of model using ar function
#------------------------------------------------------------
md_ar <- ar(y_ar2)
md_ar

# We can use "arima" function for estimate ar model:
md_ar <- arima(y_ar2, order = c(2,0,0), method = "ML")
md_ar

#------------------------------------------------------------
# Unit root tests: AR(2) model
#-----------------------------------------------------------
# Augmented Dickey-Fuller Test
adf.test(y_ar2)

# Phillips-Perron Unit Root Test
pp.test(y_ar2)

# KPSS test
kpss.test(y_ar2)

#-----------------------------------------------------------
# ACF and PACF: AR(2) model
#-----------------------------------------------------------
par(mfrow=c(1,2))
acf(y_ar2)
pacf(y_ar2)


#===========================================================
# Movil Average models
#============================================================

#-------------------------------------------------------------
# Simulated data for MA(2) model with parameters 0.5 and -0.3
#--------------------------------------------------------------
polyroot(c(1, -0.5, 0.3))

set.seed(12345)
y_ma2 <- arima.sim(model = list(order = c(0, 0, 2), ma = c(0.5, -0.3)), n = 500)

par(mfrow=c(1,1))
plot(y_ma2, xlab = "Time", ylab = "y(t) = 0.5e(t) - 0.3e(t)")

#----------------------------------------------------------
# estimation of model using arima function
#----------------------------------------------------------
md_ma <- arima(y_ma2, order = c(0,0,2), method = "ML")
md_ma

#-------------------------------------------------------------
# Unit root tests 
#-------------------------------------------------------------
# Augmented Dickey-Fuller Test
adf.test(y_ma2)

# Phillips-Perron Unit Root Test
pp.test(y_ma2)

# KPSS test
kpss.test(y_ma2)

#---------------------------------------------------------
# ACF and PACF: MA(2) model
#---------------------------------------------------------
par(mfrow=c(1,2))
acf(y_ma2)
pacf(y_ma2)

#-----------------------------------------------------------
# Simulation of ACF for MA Model: Assignment
#------------------------------------------------------------

#------------------------------------------------------
# What is the best model?
#------------------------------------------------------
arima(y_ma2, order = c(0, 0, 1), include.mean = FALSE)
arima(y_ma2, order = c(0, 0, 2), include.mean = FALSE)
arima(y_ma2, order = c(0, 0, 3), include.mean = FALSE)

AIC_ma <- NULL
AIC_ma[1] <- AIC(arima(y_ma2, order = c(0, 0, 1), include.mean = FALSE))
AIC_ma[2] <- AIC(arima(y_ma2, order = c(0, 0, 2), include.mean = FALSE))
AIC_ma[3] <- AIC(arima(y_ma2, order = c(0, 0, 3), include.mean = FALSE))

AIC_ma
min(AIC_ma)

#=============================================================
# Simulated data for ARMA(1,2) model 
#=============================================================
set.seed(12345)
sim_arma <- arima.sim(model = list(order(1,0,2), ar = c(0.7), ma = c(0.5,-0.3)),
                  n = 500)

par(mfrow=c(1,1))
plot(sim_arma, ylab = "Value")

#------------------------------------------------------------
# Unit root tests 
#-----------------------------------------------------------
# Augmented Dickey-Fuller Test
adf.test(sim_arma)

# Phillips-Perron Unit Root Test
pp.test(sim_arma)

# KPSS test
kpss.test(sim_arma)

#--------------------------------------------------------
# Estimation
#--------------------------------------------------------
arma_md <- arima(sim_arma, order = c(1,0,2))
arma_md

#---------------------------------------------------------
# ACF and PACF
#---------------------------------------------------------
par(mfrow=c(1,2))
acf(sim_arma)
pacf(sim_arma)


#----------------------------------------------------------
# Manual tuning of the ARMA model
#----------------------------------------------------------
AIC_model <- NULL
p=3  # Order of AR that you want check
q=3  # Order of MA that you want check


# The vector AIC_model will have the AIC value of each model
# As p=0,1,2,3 and q=0,1,2,3, we have 4*4-1 = 15 cases. 
for(i in 1:p){
  for(j in 1:q){
    AIC_model[j] <- AIC(arima(sim_arma, order = c(0, 0, j), include.mean = FALSE))
    AIC_model[i+p] <- AIC(arima(sim_arma, order = c(i, 0, 0), include.mean = FALSE))
    AIC_model[i+2*p] <- AIC(arima(sim_arma, order = c(i, 0, 1), include.mean = FALSE))
    AIC_model[i+3*p] <- AIC(arima(sim_arma, order = c(i, 0, 2), include.mean = FALSE))
    AIC_model[i+4*p] <- AIC(arima(sim_arma, order = c(i, 0, 3), include.mean = FALSE))
  }
}

# k is the position of vector AIC_model with the minimum value
k <- which(AIC_model == min(AIC_model))

# 
if(k > p ) {
  Q <- k %/% p
  r <- k %% p
  if (r == 0){
    p_opt <- p
    q_opt <- Q - 2
  }else {
    p_opt <- r
    q_opt <- Q - 1
  }
  
} else {
  p_opt <- 0
  q_opt <- k
}

# the p and q values for model with minimum AIC.
p_opt
q_opt

#----------------------------------------------------------------------
# Estimate the ARIMA model with p_opt, q_opt and d selected 
#-----------------------------------------------------------------------
# Note: The use of the AIC or BIC score as a model selection criterion 
# does not guarantee that the selected model (by either AIC or BIC score) 
# does not violate the model assumptions.
#-----------------------------------------------------------------------
fit_arma <- arima(sim_arma, order = c(p_opt, 0, q_opt), include.mean = FALSE)
fit_arma

#-----------------------------------------------------------------------
# Ljung-Box test confirms that there is no autocorrelation
# left on the residuals—with a p-value of 0.62, we cannot reject 
# the null hypothesis that the residuals are white noise.
#---------------------------------------------------------------------
checkresiduals(fit_arma)

#------------------------------------------------------
# What is the best model? Choice the model with minimum AIC or BIC
#------------------------------------------------------
p_prop <- 5
q_prop <- 5

table_aic<-matrix(0,p_prop+1,q_prop+1)
table_bic <- matrix(0,p_prop+1,q_prop+1)

for (i in 0:p_prop) for (j in 0:q_prop) {
  table_aic[i+1,j+1]<-AIC(arima(sim_arma, order=c(i,0,j)))
  table_bic[i+1,j+1]<-BIC(arima(sim_arma, order=c(i,0,j)))
}
table_aic
table_bic

order_aic = which(table_aic == min(table_aic), arr.ind=TRUE)
order_bic = which(table_bic == min(table_bic), arr.ind=TRUE)

p_aic <- order_aic[1]-1
q_aic <- order_aic[2]-1

p_bic <- order_bic[1]-1
q_bic <- order_bic[2]-1

# the p and q values for model with minimum AIC.
p_aic
q_aic


# the p and q values for model with minimum BIC.
p_bic
q_bic

#----------------------------------------------------------
# What is the best model? Another approach
#---------------------------------------------------------
best_order <- c(0, 0, 0)
best_bic <- Inf
 for (i in 0:5) for (j in 0:5) {
  fit_bic <- BIC(arima(sim_arma, order = c(i, 0, j)))
  if (fit_bic < best_bic) {
    best_order <- c(i, 0, j)
    best_bic <- fit_bic
  }
 }
best_order
best_bic

#------------------------------------------------------------------
# Estimate the ARIMA model with p_opt, q_opt and d selected 
#------------------------------------------------------------------
fit_arma <- arima(sim_arma, order = c(p_bic, 0, q_bic), include.mean = FALSE)
fit_arma
fit_arma$sigma2


#--------------------------------------------------------------
# Plotting the characteristic roots
#--------------------------------------------------------------
autoplot(fit_arma)

#--------------------------------------------------------
# Forecasting AR, MA, and ARMA models
#------------------------------------------------------------
# Next 12 forecasted values 
forecastedValues_0 <- forecast(fit_arma, 12)
forecastedValues_0

par(mfrow=c(1,1))
plot(forecastedValues_0, main = "Graph with forecasting",
     col.main = "darkgreen")



