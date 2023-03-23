####Libraries####
library(aTSA)
library(tseries)
library(MASS)
library(forecast)
library(strucchange)
library(readxl)
library(vars)
library(rugarch)

####Datos####
Precio_oro <- read_excel("Desktop/Semestre VIII/Series de tiempo/Precio_oro.xlsx")
Precio_SP500 <- read_excel("Desktop/Semestre VIII/Series de tiempo/Precio_S&P500.xlsx")

####Análisis inicial de los datos####
dt <- Precio_oro[,'PX_LAST']
# Datos a serie de tiempo
dt <- ts(dt,start = c(2019,1),frequency = 261)
# Comportamiento de los precios
plot(dt)
# Auto-correlaciones
acf(dt,lag.max = 261)
# Auto-correlaciones parciales
pacf(dt)
# Diferencias de la serie
difdt<-diff(dt)
# Plot de las diferencias
plot(difdt)

#### Hipótesis 1: Esatcionalidad####
# Dickey-Fuller a la serie
adf.test(dt)
# KPSS a la serie
kpss.test(dt)
# Dickey-Fuller a las diferencias
adf.test(difdt)
# KPSS a las diferencias
kpss.test(difdt)
 # falta agregarlo al latex
####Hipótesis 2: Quiebre estructural####
# Prueba de Chow
oro_bp <- breakpoints(dt~1)
# Plottin structural breaks
oro_ci <- confint(oro_bp)
plot(dt)
lines(oro_bp)
lines(oro_ci)

####Hipótesis 3: Comportamiento "jumpy"####

dist=2*sd(difdt)
mean=mean(difdt)
# Histograma de las diferencias con dos desviaciones estandar a la media
hist(difdt)
abline(v=mean+dist,col="blue")
abline(v=mean-dist,col="blue")

# Plot de las diferencias con dos desviaciones estandar a la media
plot(difdt)
abline(h=mean+dist,col="blue")
abline(h=mean-dist,col="blue")

fuera=c()
pos=c()
cont=1
for (i in difdt)
{
  if(abs(i)>mean+dist)
  {
    fuera=c(fuera,i)
    pos=c(pos,cont)
  }
  cont=cont+1
}

length(pos)/length(difdt)
####Hipótesis 4: Prima por riesgo####
pr1=ts(dt[1:420],frequency = 5)
pr2=ts(dt[420:length(dt)],frequency = 5)
dec1=decompose(pr1)
dec2=decompose(pr2)
plot(dec1)
plot(dec2)
plot(dec1$trend)
plot(dec2$trend)

####Hipótesis 5: Relación S&P550 con el oro####
t=decompose(dt)
dt2=c()
dt3=c()
for (i in 1:length(Precio_SP500$PX_LAST)) {
  for (j in 1:length(Precio_oro$PX_LAST))
    if(as.Date(Precio_SP500$Dates[i])==as.Date(Precio_oro$Dates[j]))
    {
      dt2=c(dt2,Precio_SP500$PX_LAST[i])
      dt3=c(dt3,Precio_oro$PX_LAST[i])
      break
    }
  else
  {
    
  }
}
series=data.frame(diff(dt2), diff(dt3))

aux = data.frame(dt2, dt3)
ts.plot(aux, col=c(1, 2))
legend("topleft", legend = c("S&P500", "Oro"), col = 1:2, lty = 1)
VARselect(series,lag.max=11,type="const")
modelo<-VAR((series),p=3,type=c("const"))

####Hipótesis 6: Efecto de apalancamiento####
plot(Precio_oro$PX_LAST, type="l")
abline(v=420, col=2)
pr1=dt[1:420]
pr2=dt[420:length(dt)]

#pr1 <- ts(dt,start = c(2019,2),frequency = 261)#datos a serie de tiempo

auto.arima(pr1)
difpr1=diff(pr1)
auto.arima(difpr1)

auto.arima(pr2)
difpr2=diff(pr2)
auto.arima(difpr2)
plot(ts(difpr1))
plot(ts(difpr2))

rb=ugarchspec(variance.model = list(model="gjrGARCH",garchOrder=c(1,1)),mean.model = list(armaOrder=c(2,2)))
rbf=ugarchfit(rb,data=pr1)
rbf

rb2=ugarchspec(variance.model = list(model="gjrGARCH",garchOrder=c(1,1)),mean.model = list(armaOrder=c(4,3)))
rbf2=ugarchfit(rb2,data=pr2)
rbf2