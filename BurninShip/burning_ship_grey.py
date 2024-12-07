from __future__ import print_function
from numba import cuda
import numpy as np
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
            zx, zy = zx * zx - zy * zy + cx, 2 * zx * zy + cy
            if zx * zx + zy * zy >= 4.0:
                burningship[y, x] = 255 - int(n * 255 / iterations)  # Sfumatura per l'esterno
                return

        burningship[y, x] = 200  # Grigio chiaro per l'interno del frattale

def generate_burningship(xres, yres, iterations, x_min, x_max, y_min, y_max):
    # Allocazione della memoria per l'insieme di Burning Ship
    burningship = np.zeros((yres, xres), dtype=np.uint8)

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

    return burningship

# Parametri di risoluzione e iterazioni
xres, yres = 12800, 9600  # Risoluzione ridotta per test più veloce
iterations = 1000

# Coordinate del piano complesso per il Burning Ship
x_min, x_max = -1.8, -1.7
y_min, y_max = -0.08, 0.025

# Generazione dell'immagine del frattale Burning Ship
burningship = generate_burningship(xres, yres, iterations, x_min, x_max, y_min, y_max)

# Salvataggio dell'immagine
filename = 'burning_ship.png'
print("Saving image to", filename)
img = im.fromarray(burningship, mode='L')
img.save(filename)