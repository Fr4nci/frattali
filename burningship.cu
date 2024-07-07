#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <png.h>
#include <fstream>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n", __FILE__, __LINE__); \
    return EXIT_FAILURE;}} while(0)

// Kernel CUDA per calcolare il frattale Burning Ship con color smoothing
__global__ void create_fractal(int xres, int yres, int iterations, float xmin, float xmax, float ymin, float ymax, float* smooth_vals) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < xres && y < yres) {
        float cx = xmin + x * (xmax - xmin) / xres; // Mappatura della coordinata x nel piano complesso
        float cy = ymin + y * (ymax - ymin) / yres; // Mappatura della coordinata y nel piano complesso
        float zx = 0.0f, zy = 0.0f;
        int n;
        for (n = 0; n < iterations; ++n) {
            float x_temp = zx * zx - zy * zy + cx;
            zy = fabs(2.0f * zx * zy) + cy;
            zx = fabs(x_temp);
            if (zx * zx + zy * zy > 4.0f)
                break;
        }
        if (n < iterations) {
            float log_zn = log(zx * zx + zy * zy) / 2.0f;
            float nu = log(log_zn / log(2.0f)) / log(2.0f);
            smooth_vals[y * xres + x] = n + 1 - nu;
        } else {
            smooth_vals[y * xres + x] = n;
        }
    }
}

// Funzione per mappare i valori smooth in colori RGB
void map_to_color(float smooth_val, int iterations, png_bytep color) {
    float t = smooth_val / iterations;
    int r = (int)(9 * (1 - t) * t * t * t * 255);
    int g = (int)(15 * (1 - t) * (1 - t) * t * t * 255);
    int b = (int)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
    color[0] = (png_byte)r;
    color[1] = (png_byte)g;
    color[2] = (png_byte)b;
}

// Funzione per salvare l'immagine PNG
void save_png(const char* filename, int xres, int yres, float* smooth_vals, int iterations) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        std::cerr << "Errore: Impossibile aprire il file " << filename << " per la scrittura." << std::endl;
        return;
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        std::cerr << "Errore: png_create_write_struct fallito." << std::endl;
        fclose(fp);
        return;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        std::cerr << "Errore: png_create_info_struct fallito." << std::endl;
        png_destroy_write_struct(&png_ptr, NULL);
        fclose(fp);
        return;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        std::cerr << "Errore durante la scrittura del file PNG." << std::endl;
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return;
    }

    png_init_io(png_ptr, fp);

    // Impostazioni PNG
    png_set_IHDR(png_ptr, info_ptr, xres, yres, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    png_bytep row_pointers[yres];
    png_byte *image_data = (png_byte *)malloc(3 * xres * yres * sizeof(png_byte));
    for (int y = 0; y < yres; ++y) {
        for (int x = 0; x < xres; ++x) {
            png_bytep color = &image_data[(y * xres + x) * 3];
            map_to_color(smooth_vals[y * xres + x], iterations, color);
        }
        row_pointers[y] = &image_data[y * xres * 3];
    }

    // Scrittura dell'immagine
    png_set_rows(png_ptr, info_ptr, row_pointers);
    png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

    // Pulizia
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    free(image_data);

    std::cout << "Immagine salvata correttamente come " << filename << std::endl;
}

int main() {
    // Parametri
    int xres = 12800;
    int yres = 9600;
    int iterations = 100000;

    // Limiti del piano complesso per lo zoom
    float xmin = -2.5f, xmax = -2.0f;
    float ymin = -2.0f, ymax = 0.0f;

    // Allocazione memoria per i valori smooth
    size_t size = xres * yres * sizeof(float);
    float* smooth_vals;
    CUDA_CALL(cudaMallocManaged(&smooth_vals, size));

    // Dimensioni dei blocchi e delle griglie CUDA
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((xres + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (yres + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Esecuzione del kernel
    create_fractal<<<numBlocks, threadsPerBlock>>>(xres, yres, iterations, xmin, xmax, ymin, ymax, smooth_vals);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    // Salvataggio dell'immagine PNG
    save_png("burning_ship.png", xres, yres, smooth_vals, iterations);

    // Pulizia della memoria
    CUDA_CALL(cudaFree(smooth_vals));

    return 0;
}

