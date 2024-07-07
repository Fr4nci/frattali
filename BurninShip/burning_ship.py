from __future__ import print_function
from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as im
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
xres, yres = 12800, 9600  # Ridotto per una visualizzazione più veloce
iterations = 10000

# Coordinate iniziali del piano complesso
x_min, x_max = -2.0, 1.0 # -1.8, -1.7
y_min, y_max = -2.0, 2.0 # -0.08, 0.025

# Generazione iniziale del frattale Burning Ship
burningship_colored = generate_burningship(xres, yres, iterations, x_min, x_max, y_min, y_max)

# Salvataggio dell'immagine
filename = 'burning_ship.png'
print("saving image to", filename)
img = im.fromarray(burningship_colored)
img.save(filename)
