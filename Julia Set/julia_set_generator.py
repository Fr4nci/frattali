from __future__ import print_function
from numba import cuda
import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt

@cuda.jit
def create_julia(xres, yres, iterations, mandelbrot, c):
    x, y = cuda.grid(2)

    if x < xres and y < yres:
        zx = -2.0 + x * 3.0 / xres
        zy = -1.0 + y * 2.0 / yres
        z = complex(zx, zy)

        for n in range(iterations):
            z = z * z + c
            if (z.real * z.real + z.imag * z.imag) >= 4:
                break

        mandelbrot[y, x] = n

# Parametri di risoluzione e iterazioni
xres, yres = 12800, 9600  # Aumento della risoluzione
iterations = 10000

# Parametro c per il set di Julia (puoi cambiarlo per ottenere diverse forme)
c = complex(0.285,0.013)

# Allocazione della memoria per l'insieme di Julia
mandelbrot = np.zeros((yres, xres), dtype=np.int32)

# Copia della memoria sul dispositivo
mandelbrot_device = cuda.to_device(mandelbrot)

# Definizione delle dimensioni dei blocchi e delle griglie
threadsperblock = (32, 32)
blockspergrid_x = int(np.ceil(xres / threadsperblock[0]))
blockspergrid_y = int(np.ceil(yres / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

# Esecuzione del kernel
create_julia[blockspergrid, threadsperblock](xres, yres, iterations, mandelbrot_device, c)

# Copia del risultato di nuovo all'host
mandelbrot_device.copy_to_host(mandelbrot)

# Utilizzo di una mappa di colori logaritmica per migliorare la visibilità
mandelbrot_log = np.log(mandelbrot + 1) / np.log(iterations + 1)

# Creazione della mappa di colori
colormap = plt.get_cmap("inferno")  # Puoi scegliere diverse mappe di colori da matplotlib

# Conversione dei valori normalizzati in colori usando la colormap
mandelbrot_colored = (colormap(mandelbrot_log)[:, :, :3] * 255).astype(np.uint8)

# Salvataggio dell'immagine
filename = 'julia_colored.png'
print("saving image to", filename)
img = im.fromarray(mandelbrot_colored)
img.save(filename)
