#include "module.h"

namespace dada {
  void Module::addKernel(Kernel* kernel){
    kernels_.emplace_back(kernel);
  }
};