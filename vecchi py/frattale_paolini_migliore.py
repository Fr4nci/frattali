from __future__ import print_function
from numba import cuda
import numpy as np
from PIL import Image as im

@cuda.jit
def create_fractal(xres, yres, iterations, mandelbrot):
    x, y = cuda.grid(2)

    if x < xres and y < yres:
        cx = -2.0 + x * 3.0 / xres
        cy = -1.0 + y * 2.0 / yres
        c = complex(cx, cy)

        z = 0.0j
        for n in range(iterations):
            z = z * z + c
            if (z.real * z.real + z.imag * z.imag) >= 4:
                mandelbrot[y, x] = 255 - int(n * 255 / iterations)  # Sfumatura per l'esterno
                return

        mandelbrot[y, x] = 190  # Grigio uniforme per l'interno

# Parametri di risoluzione e iterazioni
xres, yres = 12800, 9600  # Alta risoluzione
iterations = 1000  # Numero massimo di iterazioni

# Allocazione della memoria per l'insieme di Mandelbrot
mandelbrot = np.zeros((yres, xres), dtype=np.uint8)

# Copia della memoria sul dispositivo
mandelbrot_device = cuda.to_device(mandelbrot)

# Definizione delle dimensioni dei blocchi e delle griglie
threadsperblock = (32, 32)
blockspergrid_x = int(np.ceil(xres / threadsperblock[0]))
blockspergrid_y = int(np.ceil(yres / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

# Esecuzione del kernel
print("Inizio calcolo...")
create_fractal[blockspergrid, threadsperblock](xres, yres, iterations, mandelbrot_device)

# Copia del risultato di nuovo all'host
mandelbrot_device.copy_to_host(mandelbrot)

# Salvataggio dell'immagine
filename = 'mandelbrot_gray.png'
print("Salvataggio immagine in", filename)
img = im.fromarray(mandelbrot, mode='L')  # 'L' per immagini in scala di grigi
img.save(filename)
