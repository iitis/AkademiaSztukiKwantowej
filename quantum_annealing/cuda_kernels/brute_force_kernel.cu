#include <thrust/sort.h>
#include <thrust/functional.h>

#define N 20 // ilość spinów

extern "C" {

__global__ void compute_energies(float Q[N][N], int sweep_size_exponent, float* energies, long* states, int offset){


	int ti = threadIdx.x;
	int block_idx = blockIdx.x;
	int num_threads = blockDim.x;
	int grid_size = gridDim.x;
	
	int global_idx = ti + num_threads * block_idx;
	int total_threads = num_threads * grid_size;

	int chunk_size = (1 << sweep_size_exponent);  // równoważne 2^sweep_size_exponent  

	__shared__ float sQ[N][N];

	// ładujemy wpółdzieloną pamięć

	for (int idx = ti; idx < N*N; idx = idx + num_threads){
		int i = idx / N;
		int j = idx % N;
		sQ[i][j] = Q[i][j];

	}
	__syncthreads();

	for (long idx = global_idx; idx < chunk_size; idx = idx + total_threads){
		long state_code = idx + offset * chunk_size;
		bool binary[N];
		float en = 0.0F;
		
		for (int i = 0; i < N; i++){
			binary[i] = (state_code >> i) & 1;
		}

		for (int i = 0; i < N; i++){
			if (binary[i] == 1){
				en = en + sQ[i][i];
				for (int j = i + 1; j < N; j++){

					en = en + sQ[i][j] * binary[j];

				}
			}
		}
		states[idx] = state_code;
		energies[idx] = en;
	}
}

}
