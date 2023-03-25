# TALLER 1 SERIES DE TIEMPO
###########################
# PUNTO 4
###########################
# Definir el tamaño de la muestra y el parámetro de tasa lambda
n <- 1000
lambda <- 2

# Generar una muestra aleatoria de una distribución exponencial
x <- rexp(n, lambda)

# Graficar en un plot 1x3 la serie de tiempo, su histograma y su ACF
par(mfrow = c(1, 3))
plot(x, type = "l", main = "Serie de tiempo")
hist(x, main = "Histograma")
acf(x, main = "ACF")

###########################
# PUNTO 5
###########################

# Definir la longitud de las series temporales y los valores de phi
n <- 200
phis <- c(-0.9, -0.5, 0.5, 0.9)

# Simular y graficar las series temporales para cada valor de phi
par(mfrow = c(2, 2))
for (i in 1:length(phis)) {
    phi <- phis[i]

    # Simular la serie temporal
    arima.sim(list(ar = phi), n = n) -> ts_sim

    # Estimar los parámetros del modelo AR(1)
    arima(ts_sim, order = c(1, 0, 0))$coef[1] -> phi_hat
    print(paste("El valor estimado de phi es", phi_hat))

    # Hacer predicciones para los siguientes 10 periodos
    predict(arima(ts_sim, order = c(1, 0, 0)), n.ahead = 10)$pred -> pred
    print(paste("Las predicciones para los siguientes 10 periodos son", pred))
    # Graficar la serie temporal
    plot(ts_sim, type = "l", main = paste("AR(1) con phi =", phi))
    lines(length(ts_sim):(length(ts_sim) + 9), pred, col = "red")
}

###########################
# PUNTO 6
###########################

###########################
# PUNTO 7
###########################

# Definir los parámetros del modelo ARMA(1,1)
phi <- 0.9
theta <- 0.2
sigma <- sqrt(0.25)

# Generar 10 series temporales de longitud 200
set.seed(123) # para reproducibilidad
n <- 200
series <- matrix(nrow = n, ncol = 10)
for (i in 1:10) {
    series[, i] <- arima.sim(model = list(ar = c(phi), ma = c(theta)), n = n, sd = sigma)
}

# Estimar el modelo ARMA(1,1) por máxima verosimilitud para cada serie temporal
fit <- data.frame(matrix(nrow = 10, ncol = 3))
colnames(fit) <- c("phi", "theta", "intercepto")
for (i in 1:10) {
    fit[i, ] <- arima(series[, i], order = c(1, 0, 1))$coef
}

# Comparar los estimadores con los valores reales de los parámetros
real_params <- c(phi, theta, sigma^2)
fit_df <- data.frame(real_params, t(fit))
rownames(fit_df) <- c("phi", "theta", "intercepto")
print(fit_df)

# Calcular la ACF y la PACF para las primeras 4 series
par(mfrow = c(4, 2))
for (i in 1:4) {
    acf(series[, i], main = paste("ACF Serie", i))
    pacf(series[, i], main = paste("PACF Serie", i))
}