import os
import gc
from numba import cuda, float64, config
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math

# To generate the "pink" image I created this colormap
from matplotlib.colors import LinearSegmentedColormap

colors = ["#6e6eab", "#dbdbea", "#f1a3f1"]  # viola scuro → viola medio → viola chiaro
violet_cmap = LinearSegmentedColormap.from_list("violet", colors)

# Abilita pynvjitlink se disponibile (prima dell'uso di CUDA)
# config.CUDA_ENABLE_PYNVJITLINK = 1

# Assicurati che la cartella per i frame esista
OUT_DIR = "fractal_frames"
os.makedirs(OUT_DIR, exist_ok=True)

# Costante per l'array locale nel kernel (dev'essere compile-time)
NMAX = 1000  # >= N usato nella simulazione

def generate_fractal(tmax):
    # Parametri fisici
    g = 9.8067
    m1 = 1.0
    m2 = 2.0
    l1 = 1.0
    l2 = 2.0
    miu = (m1 + m2) / m2
    r = l1 / l2
    w1s = g / l1
    w2s = g / l2

    # Griglia condizioni iniziali
    N1 = 4500
    N2 = 4500
    theta1v = np.linspace(-np.pi, np.pi, N1, dtype=np.float64)
    theta2v = np.linspace(-np.pi, np.pi, N2, dtype=np.float64)

    # Tempo e preinizializzazione
    t0 = 0.0
    tf = float(tmax)
    N = 1000  # passi temporali
    if N > NMAX:
        raise ValueError(f"N={N} supera NMAX={NMAX} per gli array locali del kernel")
    dt = (tf - t0) / (N - 1) if N > 1 else 0.0
    dts = dt * dt

    @cuda.jit(device=True, inline=True)
    def simulate_pendulum(theta1, theta2, theta10, theta20, N, dt, dts, miu, r, w1s, w2s):
        # Iniziali
        theta1[0] = theta10
        theta2[0] = theta20
        theta1[1] = theta1[0]
        theta2[1] = theta2[0]

        for i in range(1, N - 1):
            delta = theta2[i] - theta1[i]

            a00 = miu * r
            a01 = math.cos(delta)
            a10 = a01
            a11 = 1.0 / r

            vu1 = (theta1[i] - theta1[i - 1]) / dt if dt != 0.0 else 0.0
            vu2 = (theta2[i] - theta2[i - 1]) / dt if dt != 0.0 else 0.0

            b0 = vu2 * vu2 * math.sin(delta) - miu * w2s * math.sin(theta1[i])
            b1 = -vu1 * vu1 * math.sin(delta) - w1s * math.sin(theta2[i])

            detA = a00 * a11 - a01 * a10
            if abs(detA) < 1e-14:
                eps0 = 0.0
                eps1 = 0.0
            else:
                invDetA = 1.0 / detA
                eps0 = (a11 * b0 - a01 * b1) * invDetA
                eps1 = (a00 * b1 - a10 * b0) * invDetA

            theta1[i + 1] = 2.0 * theta1[i] - theta1[i - 1] + dts * eps0
            theta2[i + 1] = 2.0 * theta2[i] - theta2[i - 1] + dts * eps1

        t1f = theta1[N - 1]
        t2f = theta2[N - 1]
        return math.sin(t1f) * math.sin(t2f)

    @cuda.jit
    def fractal_kernel(theta1v, theta2v, colore, N, dt, dts, miu, r, w1s, w2s):
        j1, j2 = cuda.grid(2)
        n1 = colore.shape[0]
        n2 = colore.shape[1]
        if j1 < n1 and j2 < n2:
            theta10 = theta1v[j1]
            theta20 = theta2v[j2]
            theta1 = cuda.local.array(shape=NMAX, dtype=float64)
            theta2 = cuda.local.array(shape=NMAX, dtype=float64)
            colore[j1, j2] = simulate_pendulum(theta1, theta2, theta10, theta20, N, dt, dts, miu, r, w1s, w2s)

    # Configurazione CUDA
    threadsperblock = (16, 16)
    blockspergrid_x = (N1 + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (N2 + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Copia su device
    theta1v_d = cuda.to_device(theta1v)
    theta2v_d = cuda.to_device(theta2v)
    colore_d = cuda.device_array((N1, N2), dtype=np.float64)

    # Esecuzione del kernel
    fractal_kernel[blockspergrid, threadsperblock](theta1v_d, theta2v_d, colore_d, N, dt, dts, miu, r, w1s, w2s)

    # Ritorno su host
    colore = colore_d.copy_to_host()

    # Plot
    X, Y = np.meshgrid(theta1v, theta2v, indexing='ij')
    plt.figure(figsize=(6, 6))
    plt.pcolormesh(X, Y, colore, shading="auto", cmap="viridis")
    plt.axis("off")
    plt.gca().set_aspect("equal")
    plt.tight_layout(pad=0)
    out_path = os.path.join(OUT_DIR, f"fractal_{tmax:.3f}.png")
    # plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close("all")

    # --- CLEANUP MEMORIA ---
    try:
        colore_d.device_free()
    except Exception:
        pass
    del X, Y, colore, colore_d, theta1v, theta2v, theta1v_d, theta2v_d
    gc.collect()

    # Chiudi e resetta il contesto CUDA
    cuda.close()

    print(f"Salvato: {out_path}")

def animate_fractal():
    # This commented piece of code is the one used to create the animation by generating a lot of frames
    # t_start, t_end, n_frames = 0.0, 10.0, 1000
    # for t in np.linspace(t_start, t_end, n_frames):
    #    generate_fractal(t)
    generate_fractal(9)

if __name__ == "__main__":
    animate_fractal()
