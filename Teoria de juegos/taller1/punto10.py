import numpy
import itertools


class municipio:
    def __init__(self, d_autopista, d_bosque, tipo):
        self.d_autopista = d_autopista
        self.d_bosque = d_bosque
        self.tipo = tipo
        self.vecinos = []

    def agregar_vecino(self, vecino):
        self.vecinos.append(vecino)

    def calcular_valor(self):
        unique, counts = numpy.unique(
            [i.tipo for i in self.vecinos], return_counts=True
        )
        temp = dict(zip(unique, counts))
        tipo_vecinos = {"R": 0, "C": 0, "I": 0}
        for key in temp:
            tipo_vecinos[key] = temp[key]
        if self.tipo == "R":
            return tipo_vecinos["C"] - tipo_vecinos["I"] + self.d_bosque
        elif self.tipo == "C":
            return tipo_municipios["R"] + self.d_autopista
        elif self.tipo == "I":
            return tipo_vecinos["I"] + self.d_autopista

    def cambiar_tipo(self, tipo):
        self.tipo = tipo


#######################################################################################
a = municipio(1, 6, "")
b = municipio(3, 4, "")
c = municipio(5, 2, "")
d = municipio(1, 4, "")
e = municipio(3, 2, "")

a.agregar_vecino(b)
a.agregar_vecino(d)

b.agregar_vecino(a)
b.agregar_vecino(c)
b.agregar_vecino(e)

c.agregar_vecino(b)

d.agregar_vecino(a)
d.agregar_vecino(e)

municipios = []
municipios.extend([a, b, c, d, e])

mejor_configuracion = [a.tipo, b.tipo, c.tipo, d.tipo, e.tipo]
mejor_valor = 0

tipo_municipios = {"R": 0, "C": 0, "I": 0}

# calculate the best configuration by testing all possible combinations
perm = list(itertools.product(["R", "C", "I"], repeat=5))
for i in perm:
    for j in range(5):
        municipios[j].cambiar_tipo(i[j])
    valor = sum([municipio.calcular_valor() for municipio in municipios])
    if valor > mejor_valor:
        mejor_configuracion = [municipio.tipo for municipio in municipios]
        mejor_valor = valor

print("Mejor configuracion: ", mejor_configuracion)
print("Mejor valor: ", mejor_valor)
