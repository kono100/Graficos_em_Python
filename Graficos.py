
# Grafico 1

import matplotlib.pyplot as plt

# Dados para o gráfico
x = [1, 2, 3, 4, 5,10]
y = [2, 4, 6, 8, 10,30]

# Criar o gráfico de linha
plt.plot(x, y)

# Adicionar rótulos aos eixos
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')

# Adicionar um título ao gráfico
plt.title('Gráfico de Linha Simples')

# Exibir o gráfico
plt.show()








# ================================================================






# Grafico 2

import matplotlib.pyplot as plt
import numpy as np

# Dados para o gráfico de barras horizontal
categorias = ['Categoria A', 'Categoria B', 'Categoria C', 'Categoria D']
valores = [4, 7, 1, 5]

# Criar o gráfico de barras horizontal
plt.barh(categorias, valores, color='green')

# Adicionar rótulos aos eixos
plt.xlabel('Valores')
plt.ylabel('Categorias')

# Adicionar um título ao gráfico
plt.title('Gráfico de Barras Horizontal')

# Exibir o gráfico
plt.show()








# ================================================================






# Grafico 3
##GRAFICO DE PIZZA


import matplotlib.pyplot as plt

# Dados para o gráfico de pizza
labels = ['Categoria A', 'Categoria B', 'Categoria C', 'Categoria D']
valores = [30, 20, 25, 15]

# Cores para cada fatia
cores = ['blue', 'orange', 'green', 'red']

# Sem destaque (explode com zeros)
explode = (0, 0, 0, 0)

# Criar o gráfico de pizza
plt.pie(valores, labels=labels, colors=cores, explode=explode, autopct='%1.1f%%', startangle=140)

# Adicionar um título ao gráfico
plt.title('Gráfico de Pizza')

# Exibir o gráfico
plt.show()







# ================================================================





# Grafico 4

import matplotlib.pyplot as plt

# Dados para o gráfico de pizza
labels = ['Categoria A', 'Categoria B', 'Categoria C', 'Categoria D']
valores = [30, 20, 25, 105]

# Cores para cada fatia
cores = ['blue', 'orange', 'green', 'red']

# Encontrar o índice do maior valor
indice_maior_valor = valores.index(max(valores))

# Criar uma lista explode com zeros, exceto o índice do maior valor
explode = [0] * len(valores)
explode[indice_maior_valor] = 0.1  # Destaca apenas a fatia do maior valor

# Criar o gráfico de pizza
plt.pie(valores, labels=labels, colors=cores, explode=explode, autopct='%1.1f%%', startangle=140)

# Adicionar um título ao gráfico
plt.title('Gráfico de Pizza')

# Exibir o gráfico
plt.show()










# ================================================================







# Grafico 5

import matplotlib.pyplot as plt
import numpy as np

# Dados para o gráfico de dispersão
x = np.random.rand(50)
y = np.random.rand(50)

# Criar o gráfico de dispersão
plt.scatter(x, y, color='purple', marker='o')

# Adicionar rótulos aos eixos
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')

# Adicionar um título ao gráfico
plt.title('Gráfico de Dispersão')

# Exibir o gráfico
plt.show()








# ================================================================







# Grafico 6

import matplotlib.pyplot as plt
import numpy as np

# Dados para o gráfico de barras empilhadas
categorias = ['Categoria A', 'Categoria B', 'Categoria C']
valores1 = [4, 7, 2]
valores2 = [2, 5, 3]

# Criar o gráfico de barras empilhadas
plt.bar(categorias, valores1, color='blue', label='Série 1')
plt.bar(categorias, valores2, color='orange', bottom=valores1, label='Série 2')

# Adicionar rótulos aos eixos
plt.xlabel('Categorias')
plt.ylabel('Valores')

# Adicionar um título ao gráfico
plt.title('Gráfico de Barras Empilhadas')

# Adicionar uma legenda
plt.legend()

# Exibir o gráfico
plt.show()








# ================================================================







# Grafico 7

import matplotlib.pyplot as plt
import numpy as np

# Dados para o gráfico de linha com múltiplas séries
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Criar o gráfico de linha com múltiplas séries
plt.plot(x, y1, label='Série 1 - seno', color='blue', linestyle='-', marker='o')
plt.plot(x, y2, label='Série 2 - cosseno', color='green', linestyle='--', marker='s')

# Adicionar rótulos aos eixos
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')

# Adicionar um título ao gráfico
plt.title('Gráfico de Linha com Múltiplas Séries')

# Adicionar uma legenda
plt.legend()

# Adicionar grade ao fundo do gráfico
plt.grid(True)

# Exibir o gráfico
plt.show()







# ================================================================







import matplotlib.pyplot as plt
import numpy as np

# Dados para o gráfico de área empilhada
x = np.arange(0, 5, 1)
y1 = [1, 3, 4, 2, 5]
y2 = [2, 1, 4, 4, 3]
y3 = [5, 2, 1, 3, 2]

# Criar o gráfico de área empilhada
plt.stackplot(x, y1, y2, y3, labels=['Componente 1', 'Componente 2', 'Componente 3'], colors=['purple', 'orange', 'green'])

# Adicionar rótulos aos eixos
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')

# Adicionar um título ao gráfico
plt.title('Gráfico de Área Empilhada')

# Adicionar uma legenda
plt.legend()

# Exibir o gráfico
plt.show()








# ================================================================







import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Dados para o gráfico de dispersão tridimensional
np.random.seed(42)
x = np.random.rand(50)
y = np.random.rand(50)
z = np.random.rand(50)

# Criar o gráfico de dispersão tridimensional
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='blue', marker='o')

# Adicionar rótulos aos eixos
ax.set_xlabel('Eixo X')
ax.set_ylabel('Eixo Y')
ax.set_zlabel('Eixo Z')

# Adicionar um título ao gráfico
ax.set_title('Gráfico de Dispersão Tridimensional')

# Exibir o gráfico
plt.show()







# ================================================================









import matplotlib.pyplot as plt
import numpy as np

# Dados para o gráfico de barras empilhadas horizontal
categorias = ['Categoria A', 'Categoria B', 'Categoria C', 'Categoria D']
valores1 = [4, 7, 2, 5]
valores2 = [2, 5, 3, 8]

# Criar o gráfico de barras empilhadas horizontal
fig, ax = plt.subplots()
ax.barh(categorias, valores1, color='blue', label='Série 1')
ax.barh(categorias, valores2, color='orange', left=valores1, label='Série 2')

# Adicionar rótulos aos eixos
ax.set_xlabel('Valores')
ax.set_ylabel('Categorias')

# Adicionar um título ao gráfico
ax.set_title('Gráfico de Barras Empilhadas Horizontal')

# Adicionar uma legenda
ax.legend()

# Exibir o gráfico
plt.show()







# ================================================================









import matplotlib.pyplot as plt

# Dados para o gráfico de pizza (rosca)
labels = ['Categoria A', 'Categoria B', 'Categoria C', 'Categoria D']
valores = [30, 20, 25, 15]

# Criar o gráfico de pizza (rosca)
plt.pie(valores, labels=labels, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4), colors=['blue', 'orange', 'green', 'red'])

# Adicionar um círculo no centro para criar o efeito de rosca
centro_circulo = plt.Circle((0,0),0.2,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centro_circulo)

# Adicionar um título ao gráfico
plt.title('Gráfico de Pizza (Rosca)')

# Exibir o gráfico
plt.show()








# ================================================================









import matplotlib.pyplot as plt
import numpy as np

# Dados para o gráfico de área empilhada
x = np.arange(1, 6, 1)
y1 = [1, 3, 4, 2, 5]
y2 = [2, 1, 4, 4, 3]
y3 = [5, 2, 1, 3, 2]

# Criar o gráfico de área empilhada
plt.stackplot(x, y1, y2, y3, labels=['Série 1', 'Série 2', 'Série 3'], colors=['blue', 'orange', 'green'])

# Adicionar rótulos aos eixos
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')

# Adicionar um título ao gráfico
plt.title('Gráfico de Área Empilhada')

# Adicionar uma legenda
plt.legend()

# Exibir o gráfico
plt.show()







# ================================================================







import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Criar a figura tridimensional
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Coordenadas dos átomos de hidrogênio (H) e oxigênio (O) em uma molécula de água
oxygen = np.array([0, 0, 0])
hydrogen1 = np.array([0.075, 0.586, 0])
hydrogen2 = np.array([-0.075, 0.586, 0])

# Adicionar esferas para representar átomos
ax.scatter(*oxygen, color='red', s=200, label='Oxigênio', edgecolor='black', linewidth=2)
ax.scatter(*hydrogen1, color='blue', s=100, label='Hidrogênio', edgecolor='black', linewidth=2)
ax.scatter(*hydrogen2, color='blue', s=100, edgecolor='black', linewidth=2)

# Adicionar cilindros para representar ligações
cylinder_radius = 0.03
ax.plot([oxygen[0], hydrogen1[0]], [oxygen[1], hydrogen1[1]], [oxygen[2], hydrogen1[2]], color='gray', linewidth=2)
ax.plot([oxygen[0], hydrogen2[0]], [oxygen[1], hydrogen2[1]], [oxygen[2], hydrogen2[2]], color='gray', linewidth=2)

# Adicionar linhas para representar átomos
ax.plot([oxygen[0], oxygen[0]], [oxygen[1], oxygen[1]], [oxygen[2] - cylinder_radius, oxygen[2] + cylinder_radius], color='black', linewidth=2)
ax.plot([hydrogen1[0], hydrogen1[0]], [hydrogen1[1], hydrogen1[1]], [hydrogen1[2] - cylinder_radius, hydrogen1[2] + cylinder_radius], color='black', linewidth=2)
ax.plot([hydrogen2[0], hydrogen2[0]], [hydrogen2[1], hydrogen2[1]], [hydrogen2[2] - cylinder_radius, hydrogen2[2] + cylinder_radius], color='black', linewidth=2)

# Configurar rótulos e título
ax.set_xlabel('Eixo X')
ax.set_ylabel('Eixo Y')
ax.set_zlabel('Eixo Z')
ax.set_title('Molécula de Água (H2O)')

# Adicionar uma legenda
ax.legend()

# Ajustar a visualização para uma melhor perspectiva
ax.view_init(elev=20, azim=30)

# Exibir o gráfico
plt.show()








# ================================================================







import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Criar a figura tridimensional
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Coordenadas dos átomos em uma representação simplificada
carbon = np.array([0, 0, 0])
hydrogen1 = np.array([1, 0, 0])
hydrogen2 = np.array([-1, 0, 0])
oxygen1 = np.array([0, 1, 0])
oxygen2 = np.array([0, -1, 0])
nitrogen = np.array([0, 0, 1])
phosphorus = np.array([0, 0, -1])

# Adicionar esferas para representar átomos
atoms = {'C': carbon, 'H1': hydrogen1, 'H2': hydrogen2, 'O1': oxygen1, 'O2': oxygen2, 'N': nitrogen, 'P': phosphorus}
for atom, coords in atoms.items():
    ax.scatter(*coords, label=atom, s=150, edgecolor='black', linewidth=2)

# Adicionar linhas para representar ligações
ax.plot([carbon[0], hydrogen1[0]], [carbon[1], hydrogen1[1]], [carbon[2], hydrogen1[2]], color='gray', linewidth=2)
ax.plot([carbon[0], hydrogen2[0]], [carbon[1], hydrogen2[1]], [carbon[2], hydrogen2[2]], color='gray', linewidth=2)
ax.plot([carbon[0], oxygen1[0]], [carbon[1], oxygen1[1]], [carbon[2], oxygen1[2]], color='gray', linewidth=2)
ax.plot([carbon[0], oxygen2[0]], [carbon[1], oxygen2[1]], [carbon[2], oxygen2[2]], color='gray', linewidth=2)
ax.plot([carbon[0], nitrogen[0]], [carbon[1], nitrogen[1]], [carbon[2], nitrogen[2]], color='gray', linewidth=2)
ax.plot([carbon[0], phosphorus[0]], [carbon[1], phosphorus[1]], [carbon[2], phosphorus[2]], color='gray', linewidth=2)

# Configurar rótulos e título
ax.set_xlabel('Eixo X')
ax.set_ylabel('Eixo Y')
ax.set_zlabel('Eixo Z')
ax.set_title('Molécula de Leite (Simplificada)')

# Adicionar uma legenda
ax.legend()

# Ajustar a visualização para uma melhor perspectiva
ax.view_init(elev=20, azim=30)

# Exibir o gráfico
plt.show()








