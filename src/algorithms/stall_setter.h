#pragma once

#include "src/core/kernel.h"
#include "src/algorithms/cfg.h"

using namespace std;

namespace dada {
  class StallSetter {
    private:
      CFG const* cfg_ = nullptr;
      Kernel const* kernel_ = nullptr;
      int arch_ = 0;
      
    public:
      std::vector<int> stalls_;

    public:
      StallSetter(Kernel const* kernel, CFG const* cfg, int arch);
      void set();
      std::vector<int> const& getStalls() const;
  };
}