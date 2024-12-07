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
                mandelbrot[y, x] = 255 - int(n * 255 / iterations)  # Sfumatura per l'esterno
                return

        mandelbrot[y, x] = 255 # Grigio uniforme per l'interno
                
# Parametri di risoluzione e iterazioni
xres, yres = 12800, 9600  # Risoluzione ridotta per test più veloce
iterations = 10000

# Parametro c per il set di Julia (puoi cambiarlo per ottenere diverse forme)
c = complex((-1)*0.7269, 0.1889) # complex(0.285, 0.013)

# Allocazione della memoria per l'insieme di Julia
mandelbrot = np.zeros((yres, xres), dtype=np.uint8)  # Cambiato a uint8 per immagine in scala di grigi

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

# Salvataggio dell'immagine
filename = 'julia_colored.png'
print("Saving image to", filename)
img = im.fromarray(mandelbrot, mode='F')
img.save(filename)
