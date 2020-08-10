#pragma once

#include <vector>
#include <memory>

#include "kernel.h"

namespace dada{
  class Module {
    public:
      std::vector<std::unique_ptr<Kernel>> kernels_;
    public:
      void addKernel(Kernel* kernel);
  };
} // namespace dada