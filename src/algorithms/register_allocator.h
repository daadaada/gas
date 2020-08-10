#pragma once

#include <map>
#include <vector>
#include <string>

#include "cfg.h"
#include "src/core/kernel.h"

namespace dada {
  struct PhysReg {
    int index = 0; // 0...224
    int width = 0; // 1, 2, 4
  };

  class RegisterAllocator {
    public:
      int max_registers_ = 255;
      int max_predicate_registers_ = 7;
      Kernel const* kernel_ = nullptr;
      CFG const* cfg_ = nullptr;

      std::map<std::pair<std::string, int>, PhysReg> reg_alloc_result_;
      std::map<std::pair<std::string, int>, int> preg_alloc_result_;
    public:
      RegisterAllocator(Kernel const* kernel, CFG const* cfg);
      virtual void allocate() = 0;

      void setMaxRegisters(int);
      void setMaxPredicateRegisters(int);

      int getRegisterCount() const;

      void printResult() const;
  };

  class LinearScanAllocator : public RegisterAllocator {
    private:
      std::map<std::pair<std::string, int>, std::pair<int, int>> computeRegisterLiveness();
      void lsAllocate();
    public:
      LinearScanAllocator(Kernel const* kernel, CFG const* cfg);
      void allocate() override;
  };

  class BinPackingAllocator : public RegisterAllocator {
    public:
      void allocate() override;
  };

  class GraphColoringAllocator : public RegisterAllocator {
    public:
      void allocate() override;
  };
}