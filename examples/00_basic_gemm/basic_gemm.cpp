
#include <iostream>
#include <sstream>
#include <vector>

#include "helper.h"

#include "cutlass/gemm/device/gemm.h"



topsError_t TopTlassSgemmNN(
    int M,
    int N,
    int K,
    float alpha,
    float const *A,
    int lda,
    float const *B,
    int ldb,
    float beta,
    float *C,
    int ldc){

    using ColumnMajor = cutlass::layout::ColumnMajor;

    using ToptlassGemm = cutlass::gemm::device::Gemm<float,
                                                      ColumnMajor,
                                                      float,
                                                      ColumnMajor,
                                                      float,
                                                      ColumnMajor>;

    ToptlassGemm gemm_operator;

    ToptlassGemm::Arguments args({M,N,K},
                                 {A,lda},
                                 {B,ldb},
                                 {C,ldc},
                                 {C,ldc},
                                 {alpha, beta});

    cutlass::Status status = gemm_operator(args);

    if (status != cutlass::Status::kSuccess) {
        return topsErrorUnknown;
    }

    // Return success, if no errors were encountered.
    return topsSuccess;

}



__global__ void InitializeMatrix_kernel(
  float *matrix,
  int rows,
  int columns,
  int seed = 0) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);

    tops::mdspan data_l3(tops::Global, &matrix[i + j * 16], 1);
    float buf_l1[1];
    buf_l1[0] = 0.5;

    tops::mdspan data_l1(tops::Private, buf_l1, 1);

    if (i < rows && j < columns) {
        // int offset = i + j * 16;

        // Generate arbitrary elements.
        // int const k = 16807;
        // int const m = 16;
        // float value = float(((offset + seed) * k % m) - m / 2);
        // (void)value;
        // buf_l1[0] = 0.5;
        tops::memcpy(ctx, data_l3, data_l1);

    }


}


/// Simple function to initialize a matrix to arbitrary small integers.
topsError_t InitializeMatrix(float *matrix, int rows, int columns, int seed = 0) {

  dim3 block(2,2);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitializeMatrix_kernel<<< grid, block >>>(matrix, rows, columns, seed);

  return topsGetLastError();
}


void print_matrix(const char* name, const float* data, int rows, int cols) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%10.4f ", data[i * cols + j]);
        }
        printf("\n");
    }
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

    TOPS_CHECK(topsDeviceSynchronize());

    TOPS_CHECK(topsMemcpy(h_result, *matrix, sizeof_matrix, topsMemcpyDeviceToHost));
    print_matrix("h_result", h_result, rows, columns);
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
    int lda = M;
    int ldb = K;
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
        TOPS_CHECK(topsFree(A));
        return result;
    }

    result = AllocateMatrix(&C_cutlass, M, N, 101);

    if (result != topsSuccess) {
        TOPS_CHECK(topsFree(A));
        TOPS_CHECK(topsFree(B));
        return result;
    }

    result = AllocateMatrix(&C_reference, M, N, 101);

    if (result != topsSuccess) {
        TOPS_CHECK(topsFree(A));
        TOPS_CHECK(topsFree(B));
        TOPS_CHECK(topsFree(C_cutlass));
        return result;
    }

    result = topsMemcpy(C_reference, C_cutlass, sizeof_C, topsMemcpyDeviceToDevice);

    if (result != topsSuccess) {
        std::cerr << "Failed to copy C_cutlass matrix to C_reference: "
        << topsGetErrorString(result) << std::endl;

        TOPS_CHECK(topsFree(C_reference));
        TOPS_CHECK(topsFree(C_cutlass));
        TOPS_CHECK(topsFree(B));
        TOPS_CHECK(topsFree(A));

        return result;
    }

    result = TopTlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc);


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

