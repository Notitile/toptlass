
#include <iostream>
#include <sstream>
#include <vector>

#include "helper.h"

// #include "tpotlass/gemm/device/gemm.h"


__global__ void InitializeMatrix_kernel(
  float *matrix,
  int rows,
  int columns,
  int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i + j * rows;

    // Generate arbitrary elements.
    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2);
    (void)value;
    // matrix[offset] = value;
  }
}


/// Simple function to initialize a matrix to arbitrary small integers.
topsError_t InitializeMatrix(float *matrix, int rows, int columns, int seed = 0) {

  dim3 block(3, 4);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitializeMatrix_kernel<<< grid, block >>>(matrix, rows, columns, seed);

  return topsGetLastError();
}


topsError_t AllocateMatrix(float **matrix,int rows,int columns,int seed=0){

    topsError_t result;

    size_t sizeof_matrix = sizeof(float) * rows * columns;

    result = topsMalloc(reinterpret_cast<void**>(matrix),sizeof_matrix);

    float* h_result = (float*)malloc(sizeof_matrix);

    if (result != topsSuccess){
        std::cerr << "Failed to allocate matrix: " << topsGetErrorString(result) << std::endl;
        return result;
    }

    result = topsMemset(*matrix,0,sizeof_matrix);

    if (result != topsSuccess) {
        std::cerr << "Failed to clear matrix device memory: "
        << topsGetErrorString(result) << std::endl;
        return result;
    }

    // Initialize matrix elements to arbitrary small integers.
    result = InitializeMatrix(*matrix, rows, columns, seed);

    topsDeviceSynchronize();

    topsMemcpy(h_result, *matrix, sizeof_matrix, topsMemcpyDeviceToHost);

    if (result != topsSuccess) {
        std::cerr << "Failed to initialize matrix: "
        << topsGetErrorString(result) << std::endl;
        return result;
    }

    return result;

}



topsError_t TestToptlassGemm(int M,int N,int K,float alpha,float beta){

    topsError_t result;

    // Compute leading dimensions for each matrix.
    // int lda = M;
    // int ldb = K;
    int ldc = M;

    // Compute size in bytes of the C matrix.
    size_t sizeof_C = sizeof(float) * ldc * N;

    // Define pointers to matrices in GPU device memory.
    float *A;
    float *B;
    float *C_cutlass;
    float *C_reference;

    TOPS_CHECK(topsInit(0));

     int devID = 0;
  
    // get device name
    TOPS_CHECK(topsSetDevice(devID));

    // topsStream_t stream;
    // TOPS_CHECK(topsStreamCreate(&stream));

    result = AllocateMatrix(&A, M, K, 0);

    if (result !=  topsSuccess) {
        return result;
    }

    result = AllocateMatrix(&B, K, N, 17);

    if (result !=  topsSuccess) {
        topsFree(A);
        return result;
    }

    result = AllocateMatrix(&C_cutlass, M, N, 101);

    if (result != topsSuccess) {
        topsFree(A);
        topsFree(B);
        return result;
    }

    result = AllocateMatrix(&C_reference, M, N, 101);

    if (result != topsSuccess) {
        topsFree(A);
        topsFree(B);
        topsFree(C_cutlass);
        return result;
    }

    result = topsMemcpy(C_reference, C_cutlass, sizeof_C, topsMemcpyDeviceToDevice);

    if (result != topsSuccess) {
        std::cerr << "Failed to copy C_cutlass matrix to C_reference: "
        << topsGetErrorString(result) << std::endl;

        topsFree(C_reference);
        topsFree(C_cutlass);
        topsFree(B);
        topsFree(A);

        return result;
    }

    return result;

}

int main(int argc,const char *arg[]){
    int problem[3] = {128,128,128};

    for (int i = 1;i<argc && i < 4;++i){
        std::stringstream ss(arg[i]);
        ss >> problem[i-1];
    }

    float scalars[2] = {1,0};

    for (int i = 4;i<argc && i<6;++i){
        std::stringstream ss(arg[i]);
        ss >> scalars[i-4];
    }

    topsError_t result = TestToptlassGemm(
        problem[0],     // GEMM M dimension
        problem[1],     // GEMM N dimension
        problem[2],     // GEMM K dimension
        scalars[0],     // alpha
        scalars[1]      // beta
    );

    if (result == topsSuccess){
        std::cout << "Passed." << std::endl;
    }

    return result == topsSuccess ? 0 : -1 ;


}

