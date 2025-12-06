

__device__ void matrix_add(float *A, float *B, float *Result){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    Result[x,y] = A[x,y] + B[x,y];
}