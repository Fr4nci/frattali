from __future__ import print_function
from numba import cuda
import numpy as np
import matplotlib.pyplot as plt

@cuda.jit
def create_burningship(xres, yres, iterations, burningship, x_min, x_max, y_min, y_max):
    x, y = cuda.grid(2)

    if x < xres and y < yres:
        cx = x_min + x * (x_max - x_min) / xres
        cy = y_min + y * (y_max - y_min) / yres
        zx, zy = 0.0, 0.0
        for n in range(iterations):
            zx, zy = abs(zx), abs(zy)
            zx, zy = zx*zx - zy*zy + cx, 2*zx*zy + cy
            if zx*zx + zy*zy >= 4.0:
                break

        burningship[y, x] = n

def generate_burningship(xres, yres, iterations, x_min, x_max, y_min, y_max):
    # Allocazione della memoria per l'insieme di Burning Ship
    burningship = np.zeros((yres, xres), dtype=np.int32)

    # Copia della memoria sul dispositivo
    burningship_device = cuda.to_device(burningship)

    # Definizione delle dimensioni dei blocchi e delle griglie
    threadsperblock = (32, 32)
    blockspergrid_x = int(np.ceil(xres / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(yres / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Esecuzione del kernel
    create_burningship[blockspergrid, threadsperblock](xres, yres, iterations, burningship_device, x_min, x_max, y_min, y_max)

    # Copia del risultato di nuovo all'host
    burningship_device.copy_to_host(burningship)

    # Utilizzo di una mappa di colori logaritmica per migliorare la visibilità
    burningship_log = np.log(burningship + 1) / np.log(iterations + 1)

    # Creazione della mappa di colori
    colormap = plt.get_cmap("inferno")  # Puoi scegliere diverse mappe di colori da matplotlib

    # Conversione dei valori normalizzati in colori usando la colormap
    burningship_colored = (colormap(burningship_log)[:, :, :3] * 255).astype(np.uint8)

    return burningship_colored

# Parametri di risoluzione e iterazioni
xres, yres = 800, 600  # Ridotto per una visualizzazione più veloce
iterations = 1000

# Coordinate iniziali del piano complesso
initial_x_min, initial_x_max = -2.0, 1.0
initial_y_min, initial_y_max = -2.0, 2.0

# Coordinate correnti del piano complesso
x_min, x_max = initial_x_min, initial_x_max
y_min, y_max = initial_y_min, initial_y_max

# Generazione iniziale del frattale Burning Ship
burningship_colored = generate_burningship(xres, yres, iterations, x_min, x_max, y_min, y_max)

# Creazione della figura e visualizzazione iniziale
fig, ax = plt.subplots()
img = ax.imshow(burningship_colored, extent=(x_min, x_max, y_min, y_max))

def update_fractal(event):
    global x_min, x_max, y_min, y_max
    if event.inaxes:
        # Ottieni i nuovi limiti delle assi
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        # Rigenera il frattale con i nuovi limiti
        new_burningship_colored = generate_burningship(xres, yres, iterations, x_min, x_max, -y_max, -y_min)

        # Aggiorna l'immagine visualizzata
        img.set_data(new_burningship_colored)
        img.set_extent((x_min, x_max, y_min, y_max))
        plt.draw()

def on_draw(event):
    global x_min, x_max, y_min, y_max
    if ax.get_xlim() == (initial_x_min, initial_x_max) and ax.get_ylim() == (initial_y_min, initial_y_max):
        reset_fractal(None)

def reset_fractal(event):
    global x_min, x_max, y_min, y_max
    # Ripristina le coordinate iniziali
    x_min, x_max = initial_x_min, initial_x_max
    y_min, y_max = initial_y_min, initial_y_max

    # Rigenera il frattale con le coordinate iniziali
    new_burningship_colored = generate_burningship(xres, yres, iterations, x_min, x_max, y_min, y_max)

    # Aggiorna l'immagine visualizzata
    img.set_data(new_burningship_colored)
    img.set_extent((x_min, x_max, y_min, y_max))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.draw()

# Collegare l'evento di aggiornamento della vista
fig.canvas.mpl_connect('button_release_event', update_fractal)
fig.canvas.mpl_connect('key_release_event', update_fractal)
fig.canvas.mpl_connect('draw_event', on_draw)

plt.show()
