#include <cuda.h>
#include <iostream>

using namespace std;

void load_n_launch(CUmodule& module, int i, int* output, int* output_d){
  CUfunction kernel;
  string kernel_name = "list" + to_string(i);
  cuModuleGetFunction(&kernel, module, kernel_name.c_str());

  void * args[] = {&output_d};
  cuLaunchKernel(kernel, 1, 1, 1, 
                 32, 1, 1, 
                 128, 0, args, 0);
  cudaDeviceSynchronize();

  cudaMemcpy(output, output_d, 32*sizeof(int), cudaMemcpyDeviceToHost);

  cout << "nops: \t" << i << ", cycles needed:\t"
       << output[0] << ".\t Cycles per loop:\t" << output[0]/512 << "\n";
}

int main() {
  int * output;
  int * output_d;

  output = (int*)malloc(32*sizeof(int));

  for(int i=0; i<32; ++i) output[i] = 0;

  cudaMalloc((void**)&output_d, 32*sizeof(int));
  
  CUmodule module;
  

  cuModuleLoad(&module, "l1-icache.cubin");
  for(int i=0; i<1024; ++i){
    load_n_launch(module, i, output, output_d);
  }  

  return 0;
  
}
