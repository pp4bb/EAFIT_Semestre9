#=======================================================
# Examples of simulation of ARIMA Models
# Time series
# Elaborated by: Manuel Correa Giraldo
#=======================================================
#-------------------------------------------------------
# Packages
#-------------------------------------------------------
install.packages("hrbrthemes")
install.packages("readxl")
install.packages("plotly")
install.packages("xts")


#-------------------------------------------------------
# Libraries
#-------------------------------------------------------
library(tseries)
library(forecast)
library(TSstudio)
library(readxl)


#-----------------------------------------------
# IPC Colombia: Data
#-----------------------------------------------
setwd("/Users/pablo4buitrago/Documents/EAFIT_Semestre9/Series de tiempo")
IPCCol <- read_excel("IPC 2000-2022.xlsx", sheet=1)

summary(IPCCol)

#------------------------------------------------
# Transform to time series format
#-------------------------------------------------
IPC.ts <- ts(IPCCol$Inflation_annual, start=c(2000, 1), end=c(2022, 11), frequency=12)
class(IPC.ts)
IPC.ts

# Check:
start(IPC.ts);end(IPC.ts);frequency(IPC.ts)

#---------------------------------------------------
# plot time series 
#--------------------------------------------------
plot(IPC.ts, xlab ="Year", ylab="IPC")

#--------------------------------------------------
# Classical decomposition
#-------------------------------------------------
IPC_decompose <- decompose(IPC.ts, type = "multiplicative")
plot(IPC_decompose)

IPC_decompose <- decompose(IPC.ts, type = "additive")
plot(IPC_decompose)

#----------------------------------------------------
# We consider a subsample of the data series
#----------------------------------------------------
IPC.sample <- window(IPC.ts, start = c(2020, 1), end = c(2022, 11))
IPC.sample

plot(IPC.sample, xlab ="Year", ylab="IPC")

#-----------------------------------------------------
# Descriptive statistics
#-----------------------------------------------------
summary(IPC.sample)

#----------------------------------------------------
# Unit root tests 
#----------------------------------------------------
# Augmented Dickey-Fuller Test
adf.test(IPC.sample)

# Phillips-Perron Unit Root Test
pp.test(IPC.sample)

# KPSS test
kpss.test(IPC.sample)

#----------------------------------------------------
# First differentiation of time series
#---------------------------------------------------
IPC.sample_d1 <- diff(IPC.sample)

#-----------------------------------------------------
# Plot
#-----------------------------------------------------
plot(IPC.sample_d1, type="l", xlab="Year", ylab="Diff IPC")

#-------------------------------------------------
# Unit root tests 
#----------------------------------------------------
# Augmented Dickey-Fuller Test
adf.test(IPC.sample_d1)

# Phillips-Perron Unit Root Test
pp.test(IPC.sample_d1)

# KPSS test
kpss.test(IPC.sample_d1)

#----------------------------------------------------
# Second differentiation of time series
#---------------------------------------------------
IPC.sample_d2 <- diff(IPC.sample_d1)

#-----------------------------------------------------
# Plot
#-----------------------------------------------------
plot(IPC.sample_d2, type="l", xlab="Year", ylab="Diff IPC")

#-------------------------------------------------
# Unit root tests 
#----------------------------------------------------
# Augmented Dickey-Fuller Test
adf.test(IPC.sample_d2)

# Phillips-Perron Unit Root Test
pp.test(IPC.sample_d2)

# KPSS test
kpss.test(IPC.sample_d2)

#----------------------------------------------------
# ACF and PACF plots
#-----------------------------------------------------
par(mfrow=c(1,2))
acf(IPC.sample_d1)
pacf(IPC.sample_d1)

#----------------------------------------------------
# ACF and PACF plots
#-----------------------------------------------------
par(mfrow=c(1,2))
acf(IPC.sample_d2)
pacf(IPC.sample_d2)

#---------------------------------------------------
# Best model
#----------------------------------------------------
best_order <- c(0, 0, 0)
best_bic <- Inf
for (i in 0:5) for (j in 0:5) {
  fit_bic <- BIC(arima(IPC.sample_d2, order = c(i, 0, j)))
  if (fit_bic < best_bic) {
    best_order <- c(i, 0, j)
    best_bic <- fit_bic
  }
}
best_order
best_bic

#-----------------------------------------------------------
# Estimation
#-----------------------------------------------------------
fit_IPC <- arima(IPC.sample_d2, order = best_order, include.mean = FALSE)
summary(fit_IPC)

fit_IPC <- arima(IPC.sample, order = c(0,2,1), include.mean = FALSE)
summary(fit_IPC)

#------------------------------------------------------------
# the residuals are white noise?
#--------------------------------------------------------------  
checkresiduals(fit_IPC)

#--------------------------------------------------------------
# Plotting the characteristic roots
#--------------------------------------------------------------
autoplot(fit_IPC)


#--------------------------------------------------------
# Forecasting 
#------------------------------------------------------------
# Next 12 forecasted values (you choose this value) 
forecastedValues_0 <- forecast(fit_IPC, 12)
forecastedValues_0 # print forecasted values

plot(forecastedValues_0, main = "Graph with forecasting",
     col.main = "darkgreen")

#============================================================
# auto.arima function
#============================================================
fit_1 <- auto.arima(IPC.sample)
fit_1
# Next 12 forecasted values 
forecastedValues_1 <- forecast(fit_1, 12)
forecastedValues_1


plot(forecastedValues_1, main = "Graph with forecasting",
     col.main = "darkgreen") 


