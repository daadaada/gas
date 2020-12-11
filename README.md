## Simplifying Low-Level GPU Programming with GAS

### Requirements
* cmake >= 3.8
* g++ >= 9.2
* nvcc >= 11.0 and nvcc 10.1 (since nvcc 11.0 is incompatible with c++17)

### Target
NVIDIA GPUs, arch >= sm_70

### Example
```c++
// vector copy one element
copy(s64 dst, s64 src) {
  // declare variables
  b32 .v4 Fragment[4];
  s64 LoadPtr, StorePtr;

  // load parameters
  ldc.64 LoadPtr, src;
  ldc.64 StorePtr, dst;

  // copy
  ldg.128 Fragment[0:3], [LoadPtr];
  stg.128 [StorePtr], Fragment[0:3];
  exit;
}
```

### Build
```bash
export GAS_HOME=$PWD

mkdir build
cd build
cmake ..
make # -j6
```

### Run examples
1. benchmarks
```bash
cd examples/benchmark
# can replace cpi with L0-i-cache/L1-i-cache
cd cpi
# Set target arch
export ARCH=<arch> # e.g., export ARCH=75
make
./benchmark
```

2. hgemm (Require Turing devices)
```bash
cd examples/hgemm
# Set nvcc's and ptxas' paths
# ptxas' path is usually /usr/local/cuda-11.0/bin/ptxas
export PTXAS=<path/to/ptxas11.0> # We need ptxas 11.0 for better performance,
                                 # and the support for mma instructions
export NVCC=<path/to/nvcc10.1>   # We need nvcc 10.1 to compile c++17 code
make
./run.sh
```

3. sgemm 
```bash
cd examples/sgemm
make
./run.sh
```

### Troubleshooting
If cmake cannot find ANTRL. Copy bin/antlr-4.7.2-complete.jar to /usr/local/lib/antlr-4.7.2-complete.jar
