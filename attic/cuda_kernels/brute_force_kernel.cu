#define N 64

extern "C" {


__global__ void compute_energies(float* Q, int size,  int sweep_size_exponent, float* energies, long* states, long offset){


	int ti = threadIdx.x;
	int block_idx = blockIdx.x;
	int num_threads = blockDim.x;
	int grid_size = gridDim.x;
	
	int global_idx = ti + num_threads * block_idx;
	int total_threads = num_threads * grid_size;

	long chunk_size = (1 << sweep_size_exponent);  // równoważne 2^sweep_size_exponent  

	__shared__ float sQ[N][N];

	// ładujemy wpółdzieloną pamięć

	for (int idx = ti; idx < size*size; idx = idx + num_threads){
		int i = idx / size;
		int j = idx % size;
		sQ[i][j] = Q[i * size + j];

	}
	__syncthreads();

	for (long idx = global_idx; idx < chunk_size; idx = idx + total_threads){
		long state_code = idx + offset * chunk_size;
		float en = 0.0F;
		
		for (int i = 0; i < size; i++){
			bool bit_i = (state_code >> i) & 1;
			if (bit_i){
				en = en + sQ[i][i];
				for (int j = i + 1; j < size; j++){
					bool bit_j = (state_code >> j) & 1;
					en = en + sQ[i][j] * bit_j;

				}
			}
		}
		states[idx] = state_code;
		energies[idx] = en;
	}
}

}
