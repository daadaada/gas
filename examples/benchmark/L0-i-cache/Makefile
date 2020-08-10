ARCH = 75
NAME = l0-icache
GAS = $(GAS_HOME)/build/gas

all:
  # python codegen.py > $(NAME).dada
	$(GAS) $(NAME).dada -o $(NAME).cubin
	# nvcc main.cu -lcuda -arch=sm_$(ARCH) -o $(NAME)

clean:
	rm $(NAME).cubin $(NAME) 