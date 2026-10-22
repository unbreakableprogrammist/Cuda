#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "Błąd CUDA w pliku %s (linia %d): %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

__global__ void kernel(char* pixels,double LEFT,double BOTTOM,double RIGHT,double TOP,double SIZE,double MaxIters) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // skopiowane
    short count;
    double zr, zi, betar, betai, a, b;
    double rsquared, isquared, mr, mi, msquared;
    if (x < SIZE && y < SIZE) {
        zr = 1.0;
        zi = 0.0;
        betar = LEFT + x * (RIGHT - LEFT) / SIZE;
        betai = BOTTOM + y * (TOP - BOTTOM) / SIZE;

        a = 0.5 * (1 - betar);
        b = -0.5 * betai;
        rsquared = zr * zr;
        isquared = zi * zi;

        for (count = 0; rsquared + isquared >= 2.25/(betar*betar + betai*betai) && count < MaxIters; count++)
        {
            mr = 2*a - 1 + exp(a * zr - b * zi) * cos(a * zi + b * zr);
            mi = 2*b + exp(a * zr - b * zi) * sin(a * zi + b * zr);
            msquared = mr * mr + mi * mi;
            zr = 1 - 2 * (a * mr + b * mi)/msquared;
            zi = 2 * (a * mi - b * mr)/msquared;
            rsquared = zr * zr;
            isquared = zi * zi;
        }
        if (rsquared + isquared >= 2.25/(betar*betar+betai*betai))
            pixels[(int)(x + y*SIZE)] = '*';
        else
            pixels[(int)(x + y*SIZE)] = '.';
    }
}

int main(int argc, char** argv) {


    double LEFT, BOTTOM, RIGHT, TOP, SIZE, MaxIters;

    if ( argc==1 )
    {
        LEFT = -10.;
        BOTTOM = -10.;
        RIGHT = 10.;
        TOP = 10.;
        SIZE = 40.;
        MaxIters = 80.;
    }
    else
    {
        LEFT = atof(argv[1]);
        BOTTOM = atof(argv[2]);
        RIGHT = atof(argv[3]);
        TOP = atof(argv[4]);
        SIZE = atof(argv[5]);
        MaxIters = atof(argv[6]);
    }
    char* pixels;
    int total_pixels = (int)(SIZE*SIZE);
    CUDA_CHECK(cudaMalloc((void**)&pixels, total_pixels*sizeof(char)));

    dim3 threads_per_block(16,16);
    dim3 blocks_per_grid(3,3 );

    kernel<<<blocks_per_grid, threads_per_block>>>(pixels, LEFT, BOTTOM, RIGHT, TOP, SIZE, MaxIters);
    char *h_pixels = (char*)malloc(total_pixels * sizeof(char));
    CUDA_CHECK(cudaMemcpy(h_pixels, pixels, total_pixels * sizeof(char), cudaMemcpyDeviceToHost));

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%c", h_pixels[(int)(i*SIZE + j)]);
        }
        printf("\n");
    }
    free(h_pixels);
    CUDA_CHECK(cudaFree(pixels));
}

