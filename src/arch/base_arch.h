#pragma once

#include "src/core/instruction.h"
#include "src/algorithms/register_allocator.h"

namespace dada {
  struct CtrlLogic {
    int stalls = 1;
    int read_barrier_idx = -1;
    int write_barrier_idx = -1;
    int wait_barriers_mask = 0;
    int reuse_mask = 0; // bit mask
    bool yield = false;
  };

  class BaseArch {
    public:
      virtual bool isVariableLatency(Instruction const* instr) = 0;
      virtual int getLatency(Instruction const* instr) = 0;
      virtual int getTputLatency(Instruction const* instr1, Instruction const* instr2) = 0;
      virtual int getParameterBaseOffset() const = 0;
    public:
      virtual std::vector<uint64_t> getInstructionBinary(
        Instruction const* instr, RegisterAllocator const*, CtrlLogic const&) const = 0;
  }; // class BaseArch
} // namespace dada