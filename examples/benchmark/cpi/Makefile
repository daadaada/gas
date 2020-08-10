ARCH = 75
NAME = benchmark
GAS = $(GAS_HOME)/build/gas

all:
	$(GAS) $(NAME).dada -o $(NAME).cubin
	nvcc main.cu -lcuda -arch=sm_$(ARCH) -o $(NAME)

clean:
	rm $(NAME).cubin $(NAME) 