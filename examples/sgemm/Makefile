ARCH = 70 # modify this
NAME = sgemm
GAS = $(GAS_HOME)/build/gas

all:
	/usr/local/cuda-11.0/bin/ptxas -arch=sm_$(ARCH) sgemm-gen.ptx -o sgemm-ptx.cubin
	$(GAS) -arch sm_$(ARCH) sgemm-fused.gas -o sgemm-gas.cubin
	$(GAS) -arch sm_$(ARCH) sgemm-mimic.gas -o sgemm-gas-mimic.cubin
	/usr/local/cuda-10.1/bin/nvcc main.cu -lcuda -lcublas -arch=sm_$(ARCH) -o $(NAME)

clean:
	rm $(NAME).cubin $(NAME) 
