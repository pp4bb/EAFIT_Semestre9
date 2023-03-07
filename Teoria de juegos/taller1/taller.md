<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

# Taller 1 teoría de juegos
## Primera parte - un juego de mercado estratégico
### **a) Defina este juego en forma normal: conjunto de jugadores (𝑁), conjunto de acciones de cada jugador (𝐴<sub>𝑖,</sub>), conjunto de acciones conjuntas (𝐴), y dos instancias de la función de pagos (𝑢<sub>𝑖</sub>). Brinde una interpretación económica de dos elementos del conjunto de acciones conjuntas.**

#### Definición del juego
* N = { Empresa1 Empresa2 }
* A<sub>1</sub> = { N, V }
* A<sub>2</sub> = { P, T, R }
* A = { (N,P), (N,T), (N,R), (V,P), (V,T), (V,R) }

#### Dos instancias de la función de pagos
* u<sub>1</sub>(a<sub>1</sub>=N, a<sub>2</sub>=P) = 2
* u<sub>2</sub>(a<sub>1</sub>=V, a<sub>2</sub>=R) = 2
  
#### Interpretación de dos elementos del conuunto de acciones conjuntas
* (N,P) = Cuando la empresa 1 escoge hacer una campaña normal y la empresa 2 escoge usar periódicos, ambas empresas reciben 2000 clientes mensuales.
* (V,R) = Cuando la empresa 1 escoge hacer una campaña viral y la empresa 2 escoge usar redes sociales, la empresa 1 tendrá 0 clientes mensuales, mientras que la empresa 2 tendrá 2000 clientes mensuales.

### **b) Construya las mejores respuestas en estrategias puras de cada jugador, 𝐵𝑅𝑖(𝑎<sub>−𝑖</sub>). Brinde una interpretación económica que sintetice las mejores respuestas de cada jugador.**

#### Empresa 1
* BR<sub>1</sub>(a<sub>2</sub>=P) = V
* BR<sub>1</sub>(a<sub>2</sub>=T) = V
* BR<sub>1</sub>(a<sub>2</sub>=R) = N

Si la empresa 2 escoge usar redes sociales, la mejor respuesta de la empresa 1 será hacer una campaña normal (N). En cualquier otro caso su mejor respuesta será hacer una campaña viral (V)

#### Empresa 2
* BR<sub>2</sub>(a<sub>1</sub>=P) = T
* BR<sub>2</sub>(a<sub>1</sub>=T) = R

Si la empresa 1 escoge hacer una campaña normal, la mejor respuesta de la empresa 2 será usar la televisión (T). Si la empresa 1 escoge hacer una campaña viral, la mejor respuesta de la empresa 2 será usar las redes sociales (R).

### **c) Con base en la matriz de pagos, deduzca cuántos Equilibrios de Nash en Estrategias Puras (ENEP) tiene este juego. Brinde una interpretación económica de su resultado.**

                <Poner aquí la foto de la matriz de pagos pintada>

No hay equilibrios de Nash en estrategias puras en este juego. Ningún jugador tiene una estrategia dominante, y sus estrategias no son complementarias de tal manera que lleven a un resultado estable. Por lo tanto vale la pena buscar equilibrios de Nash en estrategias mixtas

### **d) ConstruyaysimplifiquelafuncióndeutilidadesperadadelaEmpresa1decada una de sus acciones puras. Esto es 𝑢<sub>1</sub><sup>𝐸</sup>(𝑁) y 𝑢<sub>1</sub><sup>𝐸</sup>(V).**

$$\begin{aligned}
𝑢_{1}^𝐸(𝑁) &= \alpha[2\beta+0\gamma+1(1-\beta-\gamma)]\\
&= \alpha[2\beta+1-\beta-\gamma]\\
&= \alpha[\beta+1-\gamma]
\end{aligned}$$

$$\begin{aligned}
𝑢_{1}^𝐸(V) &= (1-\alpha)[3\beta+1\gamma+0(1-\beta-\gamma)]\\
&= (1-\alpha)[3\beta + \gamma]
\end{aligned}$$

### **e)Construya y simplifique la función de utilidad esperada de la Empresa 2 de cada una de sus acciones puras. Esto es 𝑢<sub>2</sub><sup>𝐸</sup>(P), 𝑢<sub>2</sub><sup>𝐸</sup>(T) y  𝑢<sub>2</sub><sup>𝐸</sup>(R).**

$$\begin{aligned}
𝑢_{2}^𝐸(P) &= \beta[2\alpha+1(1-\alpha)]\\
&= \beta[2\alpha+1-\alpha]\\
\end{aligned}$$

$$\begin{aligned}
𝑢_{2}^𝐸(T) &= \gamma[3\alpha+0(1-\gamma)]\\
&= 3\alpha\gamma
\end{aligned}$$

$$\begin{aligned}
𝑢_{2}^𝐸(R) &= (1-\beta-\gamma)[2\alpha+2(1-\alpha)]\\
&= (1-\beta-\gamma)[2\alpha+2-2\alpha]\\
&= 2(1-\beta-\gamma)
\end{aligned}$$

### **f) Construya la correspondencia de mejor respuesta de la Empresa 1, 𝐵𝑅1(𝛽, 𝛾), con base en las funciones de utilidad esperada que usted halló en el numeral (d). Interprete brevemente la correspondencia 𝐵𝑅1(𝛽, 𝛾).**

$$BR_{1}(\beta,\gamma) = 
\begin{cases}
\alpha = 1  &\rightarrow 𝑢_{1}^𝐸(𝑁) > 𝑢_{1}^𝐸(V)\\
\alpha = 0 &\rightarrow 𝑢_{1}^𝐸(𝑁) < 𝑢_{1}^𝐸(V)\\
\alpha \in [0,1] &\rightarrow 𝑢_{1}^𝐸(𝑁) = 𝑢_{1}^𝐸(V)\\
\end{cases}$$

Luego, reemplazando obtenemos que:

$$\begin{aligned}
𝑢_{1}^𝐸(N) &> 𝑢_{1}^𝐸(V)\\
\beta+1-\gamma &> 3\beta+\gamma\\
1 &> 2(\beta+\gamma)\\
\frac{1}{2}&>\beta+\gamma\\
\end{aligned}$$

Por lo tanto 𝐵𝑅1(𝛽, 𝛾)

$$BR_{1}(\beta,\gamma) = 
\begin{cases}
\alpha = 1  &\rightarrow \beta+\gamma < \frac{1}{2}\\
\alpha = 0 &\rightarrow \beta+\gamma > \frac{1}{2}\\
\alpha \in [0,1] &\rightarrow \beta+\gamma = \frac{1}{2}\\
\end{cases}$$

### **g) Demuestre que para la Empresa 2, la campaña en Periódicos (P) nunca hará parte de una mejor respuesta. Pista: hay cuatro casos en los que se llega a una contradicción matemática cuando la campaña en Periódicos (P) hace parte de la mejor respuesta de la Empresa 2.**


        <Este es el que hay que preguntarle al profesor, por que como lo hice solo encontre una contradicción>

### **h) Halle la correspondencia de mejor respuesta de la Empresa 2, 𝐵𝑅2(𝛼), que solo considera campañas en Televisión (T) y/o Redes Sociales (R).**

|     | T   | R   |
| --- | --- | --- |
| N   | 3   | 2   |
| V   | 0   | 2   |


$$BR_{2}(\alpha) = 
\begin{cases}
\beta=0 &\gamma = 1  &\rightarrow 𝑢_{2}^𝐸(T) > 𝑢_{2}^𝐸(R)\\
\beta=0 &\gamma = 0 &\rightarrow 𝑢_{2}^𝐸(T) < 𝑢_{2}^𝐸(R)\\
\beta=0 &\gamma \in [0,1] &\rightarrow 𝑢_{2}^𝐸(T) = 𝑢_{2}^𝐸(R)\\
\end{cases}$$

Luego, reemplazando obtenemos que:

$$\begin{aligned}
𝑢_{2}^𝐸(T) &> 𝑢_{2}^𝐸(R)\\
3\alpha > 2\\
\alpha > \frac{2}{3}
\end{aligned}$$

Por lo tanto 𝐵𝑅1(𝛽, 𝛾)

$$BR_{2}(\alpha) = 
\begin{cases}
\beta=0 &\gamma = 1  &\rightarrow \alpha > \frac{2}{3}\\
\beta=0 &\gamma = 0 &\rightarrow \alpha < \frac{2}{3}\\
\beta=0 &\gamma \in [0,1] &\rightarrow \alpha = \frac{2}{3}\\
\end{cases}$$

### **i) Halle al menos un Equilibrio de Nash en Estrategias Mixtas (ENEM) en el cual la Empresa 2 decida no anunciar en Periódicos (P), es decir, 𝛽 = 0. Grafique las correspondencias de mejor respuesta de ambas empresas y brinde una interpretación económica de sus resultados.**

                            <Acá va la foto de la gráfica>

En este juego de estrategias mixtas hay un solo equilibrio de Nash, el cual se alcanza para los valores de $\gamma = 1/2$ y $\alpha = 2/3$. Esto significa que cuando la las empresas 2 y 1 asignan esos valores a sus probabilidades (respectivamente), las mejores respuestas de cada jugador coincidiran, llegando al equilibrio

                            <No sé que más decir para la interpretación económica>