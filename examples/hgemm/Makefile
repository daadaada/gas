ARCH ?= 75
NAME = hgemm
GAS = $(GAS_HOME)/build/gas
PTXAS ?= /usr/local/cuda-11.0/bin/ptxas
NVCC  ?= /usr/local/cuda-10.1/bin/nvcc # nvcc 11.0's support for C++17 is broken

all:
	$(PTXAS) -arch=sm_$(ARCH) hgemm_gen.ptx -o hgemm-ptx.cubin
	$(GAS) hgemm_fused.gas -o hgemm-gas.cubin
	$(GAS) hgemm-mimic.gas -o hgemm-gas-mimic.cubin
	$(NVCC) main.cu -lcuda -lcublas -arch=sm_$(ARCH) -o $(NAME)

clean:
	rm $(NAME).cubin $(NAME) 
