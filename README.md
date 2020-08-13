## Simplifying Low-Level GPU Programming with GAS

### Requirements
* cmake >= 3.8
* g++ >= 9.2
* nvcc >= 11.0

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
make
./benchmark
```

2. hgemm (Require Turing devices)
```bash
cd examples/hgemm
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
