#include <iostream>
#include <random>
#include <thread>
#include <chrono>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define TIMES 10

using namespace std;

template<class F, class... Args>
float cuTime(F&& f, Args&&... args){
  float time_ms;
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  for(int i=0; i<TIMES; ++i){
    f(forward<Args>(args)...);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms, start, stop);
  time_ms = time_ms / (float)TIMES;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return time_ms;
}

int main(int argc, char* argv[]){
  int m = stoi(argv[1]);
  int n = m;
  int k = m;
  float gops = (uint64_t)m*n*k*2/1024.0f/1024.0f/1024.0f;

  half* A_d, *B_d, *C_d, *A_h, *B_h;
  half* C_d_cublas;

  cudaMalloc((void**)&A_d, m*k*sizeof(half));
  cudaMalloc((void**)&B_d, n*k*sizeof(half));
  cudaMalloc((void**)&C_d, m*n*sizeof(half));
  cudaMalloc((void**)&C_d_cublas, m*n*sizeof(half));

  A_h = (half*)malloc(m*k*sizeof(half));
  B_h = (half*)malloc(n*k*sizeof(half));

  mt19937 re;
  re.seed(chrono::system_clock::now().time_since_epoch().count());
  uniform_real_distribution<float> dist(0.0f, 1.0f);

  // for(int i=0; i<m*k; ++i) A_h[i] = dist(re);
  // for(int i=0; i<n*k; ++i) B_h[i] = dist(re);
  for(int i=0; i<m*k; ++i) A_h[i] = 1.0f;
  for(int i=0; i<n*k; ++i) B_h[i] = 1.0f;

  cudaMemcpy(A_d, A_h, m*k*sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, n*k*sizeof(half), cudaMemcpyHostToDevice);

  // this_thread::sleep_for(std::chrono::milliseconds(2500));

  // cuBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

  half alpha = 1.0f;
  half beta = 0.0f;

  float cublas_time = cuTime(cublasGemmEx, 
    handle,
    CUBLAS_OP_T, CUBLAS_OP_N,
    m, n, k, 
    &alpha,
    A_d, CUDA_R_16F, k, 
    B_d, CUDA_R_16F, k, 
    &beta,
    C_d_cublas, CUDA_R_16F, m, 
    CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
  );
  float cublas_tput = gops/cublas_time;
  // cout << "cuBLAS' achieved flops: " << cublas_tput << " TFLOPS.\n";

  // ptx ver
  CUmodule module;
  CUfunction kernel;
  cuModuleLoad(&module, "hgemm-ptx.cubin");
  cuModuleGetFunction(&kernel, module, "hgemm1688_256x256_tn");

  void* args[] = {&A_d, &B_d, &C_d, &m, &n, &k};

  int num_block_a = (m+(256-1))/256;
  int num_block_b = (n+(256-1))/256;

  int num_block_x = 8;
  int num_block_y = num_block_b;
  int num_block_z = num_block_a / 8;

  cudaStream_t stream{0};
  void** extra = nullptr;

  float ptx_time = cuTime(
    cuLaunchKernel,
    kernel, 
    num_block_x, num_block_y, num_block_z, 
    32, 8, 1, 
    36*1024, stream, args, extra);

  cudaDeviceSynchronize();
  float ptx_tput = (float)gops/ptx_time;

  // GAS
  cuModuleLoad(&module, "hgemm-gas.cubin");
  cuModuleGetFunction(&kernel, module, "hgemm1688_256x256_tn");

  float gas_time = cuTime(
    cuLaunchKernel, 
    kernel, 
    num_block_x, num_block_y, num_block_z, 
    32, 8, 1, 
    36*1024, stream, args, extra);
  cudaDeviceSynchronize();
  float gas_tput = (float)gops/gas_time;

  // GAS-mimic
  cuModuleLoad(&module, "hgemm-gas-mimic.cubin");
  cuModuleGetFunction(&kernel, module, "hgemm1688_256x256_tn");

  float gas_mimic_time = cuTime(
    cuLaunchKernel, 
    kernel, 
    num_block_x, num_block_y, num_block_z, 
    32, 8, 1, 
    36*1024, stream, args, extra);
  cudaDeviceSynchronize();
  float gas_mimic_tput = (float)gops/gas_mimic_time;
  printf("%d,%.2f,%.2f,%.2f,%.2f\n", m, cublas_tput, ptx_tput, gas_tput, gas_mimic_tput);

  return 0;
}
