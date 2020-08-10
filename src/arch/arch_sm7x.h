#pragma once

#include "base_arch.h"

namespace dada {
  class SM7X : public BaseArch {
    protected:
      static inline uint32_t const parameter_base_offset = 0x160;

      static inline std::map<std::string, uint64_t> const sreg_encoding = {
        {"laneid", 0x0},
        {"threadIdx_x", 0x21}, {"threadIdx_y", 0x22}, {"threadIdx_z", 0x23},
        {"blockIdx_x", 0x25}, {"blockIdx_y", 0x26}, {"blockIdx_z", 0x27},
        {"clock_lo", 0x50}, {"srz", 0x1ff},
      };

      static inline std::map<std::string, uint64_t> const predfined_const_bank1 = {
        {"blockDim_x", 0x0}, {"blockDim_y", 0x4}, {"blockDim_z", 0x8},
        {"gridDim_x", 0xc}, {"gridDim_y", 0x10}, {"gridDim_z", 0x14},
      };
      
    protected:
      uint64_t getSm7xPredMask(Instruction const* instr, RegisterAllocator const*) const;
      uint64_t getSm7xCtrlLogicEncoding(CtrlLogic const& ctrl_logic) const;

      void setSm7xDstOpEncoding(uint64_t& first, Operand const*, RegisterAllocator const*) const;
      void setSm7xSrcOpEncoding(uint64_t& first, uint64_t& second, Operand const*, 
                                RegisterAllocator const*, int loc) const;

      void setSm7xDstPredEncoding(uint64_t& second, Operand const*, RegisterAllocator const*, int dst_loc) const;
      void setSm7xSrcPredEncoding(uint64_t& second, Operand const*, RegisterAllocator const*, int src_loc) const;

      void setSm7xDstPredEncodingDefault(uint64_t& second, int dst_loc) const;    
      void setSm7xSrcPredEncodingDefault(uint64_t& second, int src_loc) const;

      void setSm7xIIMDEncoding(uint64_t& second, Operand const*, int offset, int width) const;  
  };
} // namespace dada