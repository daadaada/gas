#include <unordered_set>

#include "reuse_setter.h"

using namespace std;

namespace dada {
  ReuseSetter::ReuseSetter(
    Kernel const* kernel, CFG const* cfg, 
    RegisterAllocator const* register_allocator, int arch
  ) : kernel_(kernel), cfg_(cfg), register_allocator_(register_allocator), arch_(arch) {}

  void ReuseSetter::set() {
    reuse_masks_.resize(kernel_->instructions_.size(), 0);
    unordered_set<Opcode> const reuseable_opcodes {
      HADD2, HMUL2, HFMA2, 
      FADD, FMUL, FFMA, MUFU,
      IADD3, IMAD, 
    };
    for(BasicBlock* bb : cfg_->basic_blocks_){
      for(int instr_idx = bb->start_; instr_idx <= bb->end_; ++instr_idx){
        auto const& instr = kernel_->instructions_[instr_idx];
        if(instr->opcode_ == FFMA)
          reuse_masks_[instr_idx] = 3;
      } // for each instruction
    } // for each basic block
  }
}