ARCH = 70
NAME = sgemm
DADA = ~/dada3/build/dada

all:
	/usr/local/cuda-11.0/bin/ptxas -arch=sm_$(ARCH) sgemm-gen.ptx -o sgemm-ptx.cubin
	$(DADA) -arch sm_$(ARCH) sgemm-fused.dada -o sgemm-gas.cubin
	$(DADA) -arch sm_$(ARCH) sgemm-mimic.dada -o sgemm-gas-mimic.cubin
	/usr/local/cuda-10.1/bin/nvcc main.cu -lcuda -lcublas -arch=sm_$(ARCH) -o $(NAME)

clean:
	rm $(NAME).cubin $(NAME) 
