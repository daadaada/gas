#pragma once

#include <vector>

#include "src/core/kernel.h"
#include "register_allocator.h"
#include "cfg.h"

namespace dada{
  class ReuseSetter{
    public:
      std::vector<int> reuse_masks_;
      Kernel const* kernel_;
      CFG const* cfg_;
      RegisterAllocator const* register_allocator_;
      int const arch_;

    public:
      ReuseSetter(Kernel const*, CFG const*, RegisterAllocator const*, int arch);
      void set();
  };
}