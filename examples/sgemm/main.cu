#include <cuda.h>
#include <cublas_v2.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>

#include "sgemm.cuh"

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

__global__
void sgemm_128x128_nt_index(
  float* A, float* B, float* C,
  int m, int n, int k,
  uint64_t* loadA, uint64_t* loadB,
  uint32_t* storeAs, uint32_t* storeBs,
  uint32_t* loadAs, uint32_t* loadBs,
  uint64_t* storeC
){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  int num_block_x = (m + (128-1))/128;

  int blockIdx_x = blockIdx.x % num_block_x;
  int blockIdx_y = blockIdx.x / num_block_x;
  int lane = threadIdx.x % 32;
  int warp = threadIdx.x / 32;

  int x_g = blockIdx_x*128 + lane*4;
  int y_g = blockIdx_y*128 + lane*4;
  int k_g = warp;

  int loadA_offset = x_g + k_g*m;
  loadA[tid] = reinterpret_cast<uint64_t>(&A[loadA_offset]);
  int loadB_offset = y_g + k_g*n;
  loadB[tid] = reinterpret_cast<uint64_t>(&B[loadB_offset]);

  int warp_x = warp % 2;
  int warp_y = warp / 2;
  // int lane_x = lane % 8;
  // int lane_y = lane / 8;
  int lane_x = lane%16/2;
  int lane_y = lane/16*2 + lane%16%2;

  storeAs[tid] = (warp*128 + lane*4)*sizeof(float);
  storeBs[tid] = (warp*128 + lane*4 + 8*128)*sizeof(float);
  loadAs[tid] = (warp_x*64 + lane_x*4)*sizeof(float);
  loadBs[tid] = (warp_y*32 + lane_y*4 + 8*128)*sizeof(float);

  int c_x = blockIdx_x*128 + warp_x*64 + lane_x*4;
  int c_y = blockIdx_y*128 + warp_y*32 + lane_y*4;
  int storeC_offset = c_x + c_y*m;
  storeC[tid] = reinterpret_cast<uint64_t>(&C[storeC_offset]);
}

int main(int argc, char* argv[]){
  int m = stoi(argv[1]);
  int n = m;
  int k = m;

  float giga_ops = static_cast<float>(m*n/1024*k*2/1024/1024);

  float* A_h = (float*)malloc(m*k*sizeof(float));
  float* B_h = (float*)malloc(n*k*sizeof(float));
  float* C_h = (float*)malloc(m*n*sizeof(float));

  mt19937 re;
  re.seed(chrono::system_clock::now().time_since_epoch().count());
  uniform_real_distribution<float> dist(0.0f, 0.5f);

  // for(int i=0; i<m*k; ++i) A_h[i] = dist(re);
  // for(int i=0; i<n*k; ++i) B_h[i] = dist(re);
  for(int i=0; i<m*k; ++i) A_h[i] = 1.0f;
  for(int i=0; i<n*k; ++i) B_h[i] = 1.0f;

  float* A_d;
  float* B_d;
  float* C_d;
  float* C_cuda;
  float* C_cublas;

  cudaMalloc((void**)&A_d, m*k*sizeof(float));
  cudaMalloc((void**)&B_d, n*k*sizeof(float));
  cudaMalloc((void**)&C_d, m*n*sizeof(float));
  cudaMalloc((void**)&C_cuda, m*n*sizeof(float));
  cudaMalloc((void**)&C_cublas, m*n*sizeof(float));

  cudaMemcpy(A_d, A_h, m*k*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, n*k*sizeof(float), cudaMemcpyHostToDevice);

  int num_block_a = (m + (128-1)) / 128;
  int num_block_b = (n + (128-1)) / 128;

  int num_threads = 32 * 8 * num_block_a * num_block_b;

  uint64_t* loadA;
  uint64_t* loadB;
  uint32_t* storeAs;
  uint32_t* storeBs;
  uint32_t* loadAs;
  uint32_t* loadBs;
  uint64_t* storeC;

  cudaMalloc((void**)&loadA, num_threads*sizeof(uint64_t));
  cudaMalloc((void**)&loadB, num_threads*sizeof(uint64_t));
  cudaMalloc((void**)&storeAs, num_threads*sizeof(uint32_t));
  cudaMalloc((void**)&storeBs, num_threads*sizeof(uint32_t));
  cudaMalloc((void**)&loadAs, num_threads*sizeof(uint32_t));
  cudaMalloc((void**)&loadBs, num_threads*sizeof(uint32_t));
  cudaMalloc((void**)&storeC, num_threads*sizeof(uint64_t));

  float cuda_time;
  cudaEvent_t cuda_start, cuda_stop;

  cudaEventCreate(&cuda_start);
  cudaEventCreate(&cuda_stop);
  cudaEventRecord(cuda_start, 0);
  for(int i=0; i<TIMES; ++i){
    sgemm_128x128_nt_cuda<<<dim3(num_block_a, num_block_b, 1), dim3(32, 8)>>>(A_d, B_d, C_cuda, m, n, k);
  }
  cudaEventRecord(cuda_stop, 0);
  cudaEventSynchronize(cuda_stop);
  cudaEventElapsedTime(&cuda_time, cuda_start, cuda_stop);
  cuda_time = cuda_time/(float)TIMES;
  float cuda_tput = giga_ops/cuda_time;

  // cuBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0f;
  float beta = 0.0f;

  float cublas_time;
  cudaEvent_t cublas_start, cublas_stop;

  cudaEventCreate(&cublas_start);
  cudaEventCreate(&cublas_stop);
  cudaEventRecord(cublas_start, 0);
  for(int i=0; i<TIMES; ++i){
    cublasSgemm(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_T,
      m, n, k, 
      &alpha,
      A_d, m,
      B_d, n,
      &beta,
      C_cublas, m
    );
  }
  cudaEventRecord(cublas_stop, 0);
  cudaEventSynchronize(cublas_stop);
  cudaEventElapsedTime(&cublas_time, cublas_start, cublas_stop);
  cublas_time = cublas_time/(float)TIMES;
  float cublas_tput = giga_ops/cublas_time;

  // ptx & gas
  cudaStream_t stream{0};
  void** extra = nullptr;

  int num_block_x = 4;
  int num_block_y = num_block_b;
  int num_block_z = num_block_a / 4;

  void* args[] = {
    &A_d, &B_d, &C_d, 
    &m, &n, &k,
  };

  // ptx's SGEMM
  CUmodule cu_module;
  CUfunction kernel;

  cuModuleLoad(&cu_module, "sgemm-ptx.cubin");
  cuModuleGetFunction(&kernel, cu_module, "sgemm_128x128_nt");

  float ptx_time = cuTime(
    cuLaunchKernel, 
    kernel, num_block_x, num_block_y, num_block_z, 
    32, 8, 1,
    16*1024, stream, args, extra
  );
  float ptx_tput = giga_ops/ptx_time;


  // gas's SGEMM
  cuModuleLoad(&cu_module, "sgemm-gas.cubin");
  cuModuleGetFunction(&kernel, cu_module, "sgemm_128x128_nt");

  float gas_time = cuTime(
    cuLaunchKernel, 
    kernel, num_block_x, num_block_y, num_block_z, 
    32, 8, 1,
    16*1024, stream, args, extra
  );
  float gas_tput = giga_ops/gas_time;

  // gas-mimic's SGEMM
  cuModuleLoad(&cu_module, "sgemm-gas.cubin");
  cuModuleGetFunction(&kernel, cu_module, "sgemm_128x128_nt");

  float gas_mimic_time = cuTime(
    cuLaunchKernel, 
    kernel, num_block_x, num_block_y, num_block_z, 
    32, 8, 1,
    16*1024, stream, args, extra
  );
  float gas_mimic_tput = giga_ops/gas_mimic_time;

  // gas-yield's SGEMM
  cuModuleLoad(&cu_module, "sgemm-yield.cubin");
  cuModuleGetFunction(&kernel, cu_module, "sgemm_128x128_nt");

  float gas_yield_time = cuTime(
    cuLaunchKernel, 
    kernel, num_block_x, num_block_y, num_block_z, 
    32, 8, 1,
    16*1024, stream, args, extra
  );
  float gas_yield_tput = giga_ops/gas_yield_time;

  printf("%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", 
          m, cuda_tput, cublas_tput, ptx_tput, gas_tput, gas_mimic_tput, gas_yield_tput);


  cudaMemcpy(C_h, C_d, m*n*sizeof(float), cudaMemcpyDeviceToHost);

  float* C_cpu = (float*)malloc(m*n*sizeof(float));
  cudaMemcpy(C_cpu, C_cublas, m*n*sizeof(float), cudaMemcpyDeviceToHost);

  // // print result.
  // int errors = 0;
  // for(int col=0; col<n; ++col){
  //   for(int row=0; row<m; ++row){
  //     float g_result = C_h[col*m + row];
  //     float c_result = C_cpu[col*m + row];
  //     float diff = C_h[col*m + row] - C_cpu[col*m + row];
  //     // printf("%.3f\t%.3f\t%i\n", C_h[col*m + row], C_cpu[col*m + row],row);
  //     // printf("%i\n", reinterpret_cast<int*>(C_h)[0]);
  //     if((g_result/c_result > 1.001 || c_result/g_result > 1.001) && errors < 10){
  //       printf("%.3f\t%.3f\t%i\t%i\n", C_h[col*m + row], C_cpu[col*m + row], row, col);
  //       errors++;
  //     }
  //   }
  // }
  // if(errors == 0){
  //   cout << "dada's SGEMM agrees with cuBLAS.\n";
  // }


}
