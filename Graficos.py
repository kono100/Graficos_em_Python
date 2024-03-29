
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





# Grafico 8

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





# Grafico 9

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







# Grafico 10

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







# Grafico 11

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






# Grafico 12


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





# Grafico 13

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









# Grafico 14

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










# ================================================================









# Grafico 15

import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

dt = 0.01
t = np.arange(0, 30, dt)
nse1 = np.random.randn(len(t))                 # white noise 1
nse2 = np.random.randn(len(t))                 # white noise 2

# Two signals with a coherent part at 10 Hz and a random part
s1 = np.sin(2 * np.pi * 10 * t) + nse1
s2 = np.sin(2 * np.pi * 10 * t) + nse2

fig, axs = plt.subplots(2, 1, layout='constrained')
axs[0].plot(t, s1, t, s2)
axs[0].set_xlim(0, 2)
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('s1 and s2')
axs[0].grid(True)

cxy, f = axs[1].cohere(s1, s2, 256, 1. / dt)
axs[1].set_ylabel('Coherence')

plt.show()










# ================================================================









# Grafico 16

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation
from matplotlib.lines import Line2D


class Scope:
    def __init__(self, ax, maxt=2, dt=0.02):
        self.ax = ax
        self.dt = dt
        self.maxt = maxt
        self.tdata = [0]
        self.ydata = [0]
        self.line = Line2D(self.tdata, self.ydata)
        self.ax.add_line(self.line)
        self.ax.set_ylim(-.1, 1.1)
        self.ax.set_xlim(0, self.maxt)

    def update(self, y):
        lastt = self.tdata[-1]
        if lastt >= self.tdata[0] + self.maxt:  # reset the arrays
            self.tdata = [self.tdata[-1]]
            self.ydata = [self.ydata[-1]]
            self.ax.set_xlim(self.tdata[0], self.tdata[0] + self.maxt)
            self.ax.figure.canvas.draw()

        # This slightly more complex calculation avoids floating-point issues
        # from just repeatedly adding `self.dt` to the previous value.
        t = self.tdata[0] + len(self.tdata) * self.dt

        self.tdata.append(t)
        self.ydata.append(y)
        self.line.set_data(self.tdata, self.ydata)
        return self.line,


def emitter(p=0.1):
    """Return a random value in [0, 1) with probability p, else 0."""
    while True:
        v = np.random.rand()
        if v > p:
            yield 0.
        else:
            yield np.random.rand()


# Fixing random state for reproducibility
np.random.seed(19680801 // 10)


fig, ax = plt.subplots()
scope = Scope(ax)

# pass a generator in "emitter" to produce data for the update func
ani = animation.FuncAnimation(fig, scope.update, emitter, interval=50,
                              blit=True, save_count=100)

plt.show()










# ================================================================









# Grafico 17

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import colors
from matplotlib.ticker import PercentFormatter

# Create a random number generator with a fixed seed for reproducibility
rng = np.random.default_rng(19680801)

# Generate a random distribution (you can replace this with your actual data)
dist1 = rng.normal(size=1000)

fig, axs = plt.subplots(1, 2, tight_layout=True)

# N is the count in each bin, bins is the lower-limit of the bin
N, bins, patches = axs[0].hist(dist1, bins=20)

# We'll color code by height, but you could use any scalar
fracs = N / N.max()

# we need to normalize the data to 0..1 for the full range of the colormap
norm = colors.Normalize(fracs.min(), fracs.max())

# Now, we'll loop through our objects and set the color of each accordingly
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)

# We can also normalize our inputs by the total number of counts
axs[1].hist(dist1, bins=20, density=True)

# Now we format the y-axis to display percentage
axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))

plt.show()










# ================================================================









# Grafico 18

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(19680801)

# example data
mu = 106  # mean of distribution
sigma = 17  # standard deviation of distribution
x = rng.normal(loc=mu, scale=sigma, size=420)

num_bins = 42

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, density=True)

# add a 'best fit' line
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
ax.plot(bins, y, '--')
ax.set_xlabel('Value')
ax.set_ylabel('Probability density')
ax.set_title('Histogram of normal distribution sample: '
             fr'$\mu={mu:.0f}$, $\sigma={sigma:.0f}$')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()










# ================================================================









# Grafico 19

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

# Fixing random state for reproducibility
np.random.seed(19680801)
# Fixing bin edges
HIST_BINS = np.linspace(-4, 4, 100)

# histogram our data with numpy
data = np.random.randn(1000)
n, _ = np.histogram(data, HIST_BINS)




def prepare_animation(bar_container):

    def animate(frame_number):
        # simulate new data coming in
        data = np.random.randn(1000)
        n, _ = np.histogram(data, HIST_BINS)
        for count, rect in zip(n, bar_container.patches):
            rect.set_height(count)
        return bar_container.patches
    return animate



    # Output generated via `matplotlib.animation.Animation.to_jshtml`.

fig, ax = plt.subplots()
_, _, bar_container = ax.hist(data, HIST_BINS, lw=1,
                              ec="yellow", fc="green", alpha=0.5)
ax.set_ylim(top=55)  # set safe limit to ensure that all data is visible.

ani = animation.FuncAnimation(fig, prepare_animation(bar_container), 50,
                              repeat=False, blit=True)
plt.show()










# ================================================================









# Grafico 20

import matplotlib.pyplot as plt
import numpy as np

# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()










# ================================================================









# Grafico 21

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.patches as patches
import matplotlib.path as path

fig, axs = plt.subplots(2)

np.random.seed(19680801)  # Fixing random state for reproducibility

# histogram our data with numpy
data = np.random.randn(1000)
n, bins = np.histogram(data, 50)

# get the corners of the rectangles for the histogram
left = bins[:-1]
right = bins[1:]
bottom = np.zeros(len(left))
top = bottom + n

# we need a (numrects x numsides x 2) numpy array for the path helper
# function to build a compound path
XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

# get the Path object
barpath = path.Path.make_compound_path_from_polys(XY)

# make a patch out of it, don't add a margin at y=0
patch = patches.PathPatch(barpath)
patch.sticky_edges.y[:] = [0]
axs[0].add_patch(patch)
axs[0].autoscale_view()

nrects = len(left)
nverts = nrects*(1+3+1)
verts = np.zeros((nverts, 2))
codes = np.ones(nverts, int) * path.Path.LINETO
codes[0::5] = path.Path.MOVETO
codes[4::5] = path.Path.CLOSEPOLY
verts[0::5, 0] = left
verts[0::5, 1] = bottom
verts[1::5, 0] = left
verts[1::5, 1] = top
verts[2::5, 0] = right
verts[2::5, 1] = top
verts[3::5, 0] = right
verts[3::5, 1] = bottom

barpath = path.Path(verts, codes)

# make a patch out of it, don't add a margin at y=0
patch = patches.PathPatch(barpath)
patch.sticky_edges.y[:] = [0]
axs[1].add_patch(patch)
axs[1].autoscale_view()

plt.show()










# ================================================================









# # Grafico 22

import matplotlib.pyplot as plt

# Dados
meses = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set","out", "nov", "dez","..." ]
indices = [1, 56, 50, 100, 5, 200,215 ,12 ,75, 90, 75, 150, 15]

# Altera a cor da linha
plt.plot(meses, indices, color="red")

# Altera o tipo de linha
plt.plot(meses, indices, linestyle="--")

# Altera os títulos dos eixos
plt.xlabel("Mês (ano 2019)")
plt.ylabel("Índice de qualidade do ar (µg/m³)")

# Cria o gráfico
plt.plot(meses, indices)

# Define os títulos dos eixos
plt.xlabel("Mês")
plt.ylabel("Índice de qualidade do ar")

# Mostra o gráfico
plt.show()












## =====================================================================










## Grafico 23

import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Função para atualizar o gráfico com base nos dados inseridos pelo usuário
def atualizar_grafico():
    notas_usuario = []
    
    for entry in entries:
        nota_texto = entry.get()
        
        # Verificar se a entrada é válida
        if nota_texto.strip():  # Verificar se a string não está vazia
            nota = float(nota_texto.replace(',', '.'))  # Aceitar vírgula ou ponto como separador decimal
            notas_usuario.append(nota)
        else:
            # Tratar caso a entrada seja vazia
            notas_usuario.append(0)

    x = list(range(1, 9))
    y = notas_usuario

    # Limpar o conteúdo antigo do gráfico
    for widget in frame_grafico.winfo_children():
        widget.destroy()

    # Criar e exibir o novo gráfico
    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o')
    ax.set_title('Notas de Matemática')
    ax.set_xlabel('Provas')
    ax.set_ylabel('Notas')

    canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Criar a janela principal
root = tk.Tk()
root.title("Visualização de Notas")

# Entradas para as notas
entries = []
for i in range(8):
    label = tk.Label(root, text=f"Nota Prova {i+1}:")
    label.pack()
    entry = tk.Entry(root)
    entry.pack()
    entries.append(entry)

# Botão para atualizar o gráfico
btn_atualizar = tk.Button(root, text="Atualizar Gráfico", command=atualizar_grafico)
btn_atualizar.pack(pady=10)

# Frame para o gráfico
frame_grafico = tk.Frame(root)
frame_grafico.pack()

# Inicializar o gráfico
atualizar_grafico()

# Iniciar o loop principal da interface gráfica
root.mainloop()
