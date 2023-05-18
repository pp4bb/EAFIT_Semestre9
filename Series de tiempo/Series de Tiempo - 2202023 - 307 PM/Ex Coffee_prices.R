# =======================================================
# Examples of simulation of ARIMA Models
# Time series
# Elaborated by: Manuel Correa Giraldo
# =======================================================
#--------------------------------------------------------
# Packages
#--------------------------------------------------------
install.packages("TSstudio")

#--------------------------------------------------------
# Libraries
#--------------------------------------------------------
library(TSstudio)
library(tseries)
library(forecast)

#--------------------------------------------------------
# Data
#--------------------------------------------------------
data("Coffee_Prices")
ts_info(Coffee_Prices)
robusta_price <- window(Coffee_Prices[, 1], start = c(1998, 1))

#--------------------------------------------------------
# Plot
#--------------------------------------------------------
plot(robusta_price, type = "l", xlab = "Year", ylab = "Price in USD")

#-----------------------------------------------------
# Descriptive statistics
#-----------------------------------------------------
summary(robusta_price)

#------------------------------------------------------------
# Box-Cox transformation
#------------------------------------------------------------
robusta_lambda <- BoxCox.lambda(robusta_price)
robusta_lambda

robusta_transform <- BoxCox(robusta_price, lambda = robusta_lambda)
BoxCox.lambda(robusta_transform)

plot(robusta_transform, type = "l", xlab = "Year", ylab = "Price in USD")

#----------------------------------------------------
# Unit root tests
#----------------------------------------------------
# Augmented Dickey-Fuller Test
adf.test(robusta_price)
adf.test(robusta_transform)

# Phillips-Perron Unit Root Test
pp.test(robusta_price)
pp.test(robusta_transform)

# KPSS test
kpss.test(robusta_price)
kpss.test(robusta_transform)

#----------------------------------------------------
# Differentiation of time series
#---------------------------------------------------
robusta_price_d1 <- diff(robusta_price)
robusta_transform_d1 <- diff(robusta_transform)

#-----------------------------------------------------
# Plot
#-----------------------------------------------------
plot(robusta_price_d1, type = "l", xlab = "Year", ylab = "Diff price in USD")
plot(robusta_transform_d1, type = "l", xlab = "Year", ylab = "Diff price in USD")

#-------------------------------------------------
# Unit root tests
#----------------------------------------------------
# Augmented Dickey-Fuller Test
adf.test(robusta_price_d1)
adf.test(robusta_transform_d1)

# Phillips-Perron Unit Root Test
pp.test(robusta_price_d1)
pp.test(robusta_transform_d1)

# KPSS test
kpss.test(robusta_price_d1)
kpss.test(robusta_transform_d1)

#----------------------------------------------------
# ACF and PACF plots: robusta_price_d1
#-----------------------------------------------------
par(mfrow = c(1, 2))
acf(robusta_price_d1)
pacf(robusta_price_d1)

#----------------------------------------------------
# ACF and PACF plots: robusta_transform_d1
#-----------------------------------------------------
par(mfrow = c(1, 2))
acf(robusta_transform_d1)
pacf(robusta_transform_d1)

#----------------------------------------------------------
# What is the best model? for robusta_price_d1
#----------------------------------------------------------
best_order1 <- c(0, 0, 0)
best_bic1 <- Inf
for (i in 0:3) {
  for (j in 0:3) {
    for (k in 0:3) {
      fit_bic1 <- AIC(arima(robusta_price, order = c(i, k, j)))
      if (fit_bic1 < best_bic1) {
        best_order1 <- c(i, k, j)
        best_bic1 <- fit_bic1
      }
    }
  }
}
best_order1
best_bic1

#----------------------------------------------------------
# What is the best model? for robusta_transform_d1
#----------------------------------------------------------
best_order2 <- c(0, 0, 0)
best_bic2 <- Inf
for (i in 0:3) {
  for (j in 0:3) {
    fit_bic2 <- BIC(arima(robusta_transform_d1, order = c(i, 0, j)))
    if (fit_bic2 < best_bic2) {
      best_order2 <- c(i, 0, j)
      best_bic2 <- fit_bic2
    }
  }
}
best_order2
best_bic2


#-----------------------------------------------------------
# Estimation
#-----------------------------------------------------------
fit_price1 <- arima(robusta_price_d1, order = best_order1, include.mean = FALSE)
summary(fit_price1)

fit_price2 <- arima(robusta_transform_d1, order = best_order2, include.mean = FALSE)
summary(fit_price2)

fit_price <- arima(robusta_price, order = c(1, 1, 0), include.mean = FALSE)
summary(fit_price)

#------------------------------------------------------------
# the residuals are white noise?
#--------------------------------------------------------------
checkresiduals(fit_price)

#--------------------------------------------------------------
# Plotting the characteristic roots
#--------------------------------------------------------------
autoplot(fit_price)

#--------------------------------------------------------
# Forecasting
#------------------------------------------------------------
# Next 12 forecasted values (you choose this value)
forecastedValues_0 <- forecast(fit_price, 12)
forecastedValues_0 # print forecasted values

plot(forecastedValues_0,
  main = "Graph with forecasting",
  col.main = "darkgreen"
)


# ============================================================
# auto.arima function
# ============================================================
fit_1 <- auto.arima(robusta_price)
fit_1
# Next 12 forecasted values
forecastedValues_1 <- forecast(fit_1, 12)
forecastedValues_1


plot(forecastedValues_1,
  main = "Graph with forecasting",
  col.main = "darkgreen"
)