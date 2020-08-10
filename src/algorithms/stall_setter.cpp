#include <iostream>

#include "stall_setter.h"
#include "src/arch/arch-common.h"

namespace dada {
  StallSetter::StallSetter(Kernel const* kernel, CFG const* cfg, int arch)
    : kernel_(kernel), cfg_(cfg), arch_(arch) {
      stalls_.resize(kernel_->instructions_.size(), 1);
    }

  void StallSetter::set() {
    unique_ptr<BaseArch> arch(makeArch(arch_));

    for(BasicBlock const* bb : cfg_->basic_blocks_){
      map<pair<string, int>, int> active_ops; // written, to be

      for(int instr_idx=bb->start_; instr_idx<=bb->end_; ++instr_idx){
        unique_ptr<Instruction> const& instr = kernel_->instructions_[instr_idx];

        int latency = arch->getLatency(instr.get());
        bool is_var_latency = arch->isVariableLatency(instr.get());

        int max_stall = 1;


        if(!is_var_latency){
          for(auto const& dst : instr->dst_operands_){
            if(dst->type_ == ID && dst->state_space_ == REG){
              for(auto dst_op : dst->getOpPairs()){
                if(active_ops.find(dst_op) != active_ops.end()){
                  active_ops[dst_op] = max(active_ops[dst_op], latency);
                } else {
                  active_ops[dst_op] = latency;
                }              
              }
            }
          }
        } else { /* instr with variable latency. needs 2 cycles for the barrier to effect */ 
          max_stall = 2;
        }

        // Check for the next instr
        if(instr_idx != bb->end_){
          auto const& next_instr = kernel_->instructions_[instr_idx+1];
          for(auto const& src : next_instr->src_operands_){
            if((src->type_ == ID || src->type_ == MEM_REF) && src->state_space_ == REG){
              for(auto src_op : src->getOpPairs()){
                if(active_ops.find(src_op) != active_ops.end()){
                  max_stall = max(max_stall, active_ops.at(src_op));
                }
              }
            }
          }
          if(next_instr->predicate_mask_){
            auto pmask = next_instr->predicate_mask_->getOpPair();
            if(active_ops.find(pmask) != active_ops.end()){
              max_stall = max(max_stall, active_ops.at(pmask));
            }
          }

          // Stall because the throughput limit
          max_stall = max(max_stall, arch->getTputLatency(instr.get(), next_instr.get()));
        } else {
          for(auto [aop, astall] : active_ops){
            max_stall = max(max_stall, astall);
          }
        }

        // TODO: move this to getTputLatency()
        if(instr->opcode_ == BAR || instr->opcode_ == BRA) max_stall = 5;

        stalls_[instr_idx] = max_stall;
        // Update active_ops
        for(auto iter = active_ops.begin(); iter != active_ops.end(); ){
          if((*iter).second <= max_stall){
            iter = active_ops.erase(iter);
          } else {
            (*iter).second -= max_stall;
            ++iter;            
          }
        }

      } // for each instr
    } // for each bb
  }

  std::vector<int> const& StallSetter::getStalls() const {
    return stalls_;
  }
} // namespace dada