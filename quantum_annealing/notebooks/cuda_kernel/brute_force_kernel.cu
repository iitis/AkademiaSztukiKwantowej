#include <thrust/sort.h>
#include <thrust/functional.h>
#include <math.h>

#define N 32 // ilość spinów
#define M 1

extern "C" {

__global__ void search(float Q[N][N], int sweep_size, float energies[M], int states[M], int a){


	int ti = threadIdx.x;
	int num_threads = blockDim.x;
	int state_code = ti + a * num_threads;

	__shared__ float sQ[N][N];

	// ładujemy wpółdzieloną pamięć

	for (int idx = ti; idx <= N*N; idx = idx + blockDim.x){
		int i = idx/N;
		int j = idx % N;

		sQ[i][j] = Q[i][j];
	}
	__syncthreads();

	


	
}

}
