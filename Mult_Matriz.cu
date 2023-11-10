#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 3
#define BLOCK_SIZE 2

__global__ void matrizMul(int *A, int *B, int *C, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < n && row < n) {
        int soma = 0;
        for (int k = 0; k < n; ++k) {
            soma += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = soma;
    }
}

int main() {
    // Declaração das matrizes (váriaveis)
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(int);

    // Aloca matrizes de host (CPU)
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);


    // Inicializa matrizes A e B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = 1;
            h_B[i * N + j] = 2;
        }
    }

    // h_B[8] = 3;

    // Aloca matrizes de dispositivo (GPU)
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copia matrizes h_A e h_B para as matrizes d_A e d_B no dispositivo
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define as dimenções do bloco e da grade
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

    // Chame o kernel matrixMul com as matrizes d_A, d_B e d_C (Escravos)
    matrizMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    
    cudaDeviceSynchronize();

    // Copie a matriz resultante d_C de volta para a matriz h_C no host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Imprima a matriz A
    printf("Matriz A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", h_A[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");

    // Imprima a matriz B
    printf("Matriz B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", h_B[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");

    // Imprima a matriz C
    printf("Resultado da multiplicacao de matrizes:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", h_C[i * N + j]);
        }
        printf("\n");
    }

    // Liberação da memória alocada no dispositivo (GPU)
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Liberação da memória alocada no host (CPU)
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
