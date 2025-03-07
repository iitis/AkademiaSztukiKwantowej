
extern "C"{ // wymagane , ponieważ c++ naturalnie dekoruje (name mangling) nazwy funkcji, przez co trudno je potem linkować

__global__ void update_x_and_wall(float* x, float* y, 
                         float a_0, float time_step, int N,
                         float* x_new, float* y_new){
    
    int ti = threadIdx.x;  // pojedyńczy element w kolumnie
    int col = blockIdx.x;  // każdy blok zajmuje się jedną kolumną (trajektorią)
    int k = blockIdx.y;  // ewentualne dodatkowe bloki na kolumne
    int blockSize = blockDim.x;  // rozmiar bloku

    // gdzie "globalnie" jesteśmy w macierzy
    int global_row = ti + k * blockSize;
    int num_cols = gridDim.x;

    if (global_row < N){
        float y_ij = y[global_row * num_cols + col];
        float temp1 = x[global_row * num_cols + col] + a_0 * y_ij * time_step;  // krok x


        // "sciana"
        if (abs(temp1) > 1){
            x_new[global_row * num_cols + col] = max(-1.0F, min(1.0F, temp1));
            y_new[global_row * num_cols + col] = 0.0F;
        } else{
            x_new[global_row * num_cols + col] = temp1;
            y_new[global_row * num_cols + col] = y_ij;
        }
    
    }
}   

__global__ void update_y(float* x, float* y, float* A, 
                         float a_0, float a_t, float c_0, float time_step, int N,
                         float* y_new){
    int ti = threadIdx.x;  // pojedyńczy element w kolumnie
    int col = blockIdx.x;  // każdy blok zajmuje się jedną kolumną (trajektorią)
    int k = blockIdx.y;  // ewentualne dodatkowe bloki na kolumne
    int blockSize = blockDim.x;  // rozmiar bloku

    // gdzie "globalnie" jesteśmy w macierzy
    int global_row = ti + k * blockSize;
    int num_cols = gridDim.x;

    float a = -1.0F * (a_0 - a_t);

    if (global_row < N){
        float y_ij = y[global_row * num_cols + col];
        float x_ij = x[global_row * num_cols + col];
        float a_ij = A[global_row * num_cols + col];
        float temp = y_ij + (a * x_ij + c_0 * a_ij) * time_step;
        y_new[global_row * num_cols + col] = temp;
    }
}

}