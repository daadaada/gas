#include <cuda.h>
#include <iostream>
#include <string>

using namespace std;

void launch_(CUfunction kernel, string name, int* output, int* output_d){
  void * args[] = {&output_d};
  cuLaunchKernel(kernel, 1, 1, 1, 
                 32, 1, 1, 
                 128, 0, args, 0);
  cudaDeviceSynchronize();

  cudaMemcpy(output, output_d, 32*sizeof(int), cudaMemcpyDeviceToHost);

  cout << "Cycles to issue 10 " << name << ":     \t"
       << output[0] << ".\n";

  for(int i=0; i<32; ++i) output[i] = 0;
}

int main() {
  int * output;
  int * output_d;

  output = (int*)malloc(32*sizeof(int));

  for(int i=0; i<32; ++i) output[i] = 0;

  cudaMalloc((void**)&output_d, 32*sizeof(int));
  
  CUmodule module;
  CUfunction ffma_kernel, fadd_kernel, hadd2_kernel, hfma2_kernel, iadd3_kernel, 
             lea_kernel, imad_kernel, imad_wide_kernel, hmma_kernel;

  cuModuleLoad(&module, "benchmark.cubin");
  cuModuleGetFunction(&ffma_kernel, module, "ffma_cpi");
  cuModuleGetFunction(&fadd_kernel, module, "fadd_cpi");
  cuModuleGetFunction(&hadd2_kernel, module, "hadd2_cpi");
  cuModuleGetFunction(&hfma2_kernel, module, "hfma2_cpi");
  cuModuleGetFunction(&iadd3_kernel, module, "iadd3_cpi");
  cuModuleGetFunction(&lea_kernel, module, "lea_cpi");
  cuModuleGetFunction(&imad_kernel, module, "imad_cpi");
  cuModuleGetFunction(&imad_wide_kernel, module, "imad_wide_cpi");
  cuModuleGetFunction(&hmma_kernel, module, "hmma_cpi");

  launch_(ffma_kernel, "ffma", output, output_d);
  launch_(fadd_kernel, "fadd", output, output_d);  
  launch_(hadd2_kernel, "hadd2", output, output_d);  
  launch_(hfma2_kernel, "hfma2", output, output_d);  
  launch_(iadd3_kernel, "iadd3", output, output_d);
  launch_(lea_kernel, "lea", output, output_d);
  launch_(imad_kernel, "imad", output, output_d);
  launch_(imad_wide_kernel, "imad.wide", output, output_d);
  launch_(hmma_kernel, "hmma.1688.f16", output, output_d);

  CUfunction ffma_hadd2_kernel, ffma_imad_kernel, ffma_lea_kernel, ffma_iadd3_kernel,
             hmma_hadd2_kernel, hmma_ffma_kernel, hmma_iadd3_kernel;
  
  cuModuleGetFunction(&ffma_hadd2_kernel, module, "ffma_hadd2");
  cuModuleGetFunction(&ffma_imad_kernel, module, "ffma_imad");
  cuModuleGetFunction(&ffma_lea_kernel, module, "ffma_lea");
  cuModuleGetFunction(&ffma_iadd3_kernel, module, "ffma_iadd3");
  cuModuleGetFunction(&hmma_hadd2_kernel, module, "hmma_hadd2");
  cuModuleGetFunction(&hmma_ffma_kernel, module, "hmma_fmma");
  cuModuleGetFunction(&hmma_iadd3_kernel, module, "hmma_iadd3");

  launch_(ffma_hadd2_kernel, "ffma_hadd2_mix", output, output_d);
  launch_(ffma_imad_kernel, "ffma_imad_mix", output, output_d);
  launch_(ffma_lea_kernel, "ffma_lea_mix", output, output_d);
  launch_(ffma_iadd3_kernel, "ffma_iadd3_mix", output, output_d);
  launch_(hmma_hadd2_kernel, "hmma_hadd2_mix", output, output_d);
  launch_(hmma_ffma_kernel, "hmma_ffma_mix", output, output_d);
  launch_(hmma_iadd3_kernel, "hmma_iadd3_mix", output, output_d);

  return 0;
  
}