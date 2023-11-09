#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define  N 4
#define BLOCK_SIZE 2

__global__ void matrixMul(int *A, int *B, int *C, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < n && row < n) {
        for (int k = 0; k < n; ++k) {
            C[row * n + col] = A[row * n + k] * B[k * n + col];
        }
    }
}

int main() {
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(int);

    // Aloca e inicializa matrizes de host (A, B e C)
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);


    // Inicializa matrizes h_A e h_B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = 2;
            h_B[i * N + j] = 1;
        }
    }

    h_A[5] = 4;

    // Aloca matrizes de dispositivo (A, B e C)
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copia matrizes h_A e h_B para as matrizes d_A e d_B no dispositivo
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

    // Chame o kernel matrixMul com as matrizes d_A, d_B e d_C
    matrixMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    // Copie a matriz resultante d_C de volta para a matriz h_C no host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Imprima a matriz h_A
    printf("Matriz A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", h_A[i * N + j]);
        }
        printf("\n");
    }

    // Imprima a matriz h_B
    printf("Matriz B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", h_B[i * N + j]);
        }
        printf("\n");
    }

    // Imprima a matriz H_C
    printf("Resultado da multiplicacao de matrizes:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", h_C[i * N + j]);
        }
        printf("\n");
    }

    // Libere a memória alocada no dispositivo
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Libere a memória alocada no host
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
