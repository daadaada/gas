#include "arch_sm7x.h"
#include <stdexcept>
#include <iostream>

using namespace std;

namespace dada {
  uint64_t SM7X::getSm7xPredMask(Instruction const* instr, RegisterAllocator const* register_allocator) const {
    if(instr->predicate_mask_){
      auto const& pred_mask = instr->predicate_mask_;
      uint64_t pred_idx = register_allocator->preg_alloc_result_.at(
        pred_mask->getOpPair());
      if(pred_mask->is_neg_) {
        pred_idx |= 1<<3;
      }
      return pred_idx << 12;
    } else {
      return 0x7 << 12;
    }
  }

  uint64_t SM7X::getSm7xCtrlLogicEncoding(CtrlLogic const& ctrl_logic) const {
    uint64_t result = 0;
    // stall | yield << 4 | wrtdb << 5 | readb << 8 | watdb << 11 | reuse << 17
    result |= (uint64_t)(ctrl_logic.stalls & 0xf) << 41;
    result |= ctrl_logic.yield? 0 : 1ULL<<45;
    result |= (uint64_t)(ctrl_logic.write_barrier_idx & 0x7) << 46;
    result |= (uint64_t)(ctrl_logic.read_barrier_idx & 0x7) << 49;
    result |= (uint64_t)(ctrl_logic.wait_barriers_mask & 0x3f) << 52;
    result |= (uint64_t)(ctrl_logic.reuse_mask & 0xf) << 58;

    return result;
  }

  void SM7X::setSm7xDstOpEncoding(uint64_t& first, Operand const* dst, 
    RegisterAllocator const* register_allocator) const {
    auto dst_op = dst->getOpPair();
    uint64_t reg_idx = 0;
    try {reg_idx = register_allocator->reg_alloc_result_.at(dst_op).index;}
    catch(exception const& e){
      cout << "In setSm7xDstOpEncoding(), src not found in register allocator.\n"
            << dst_op.first << '\t' << dst_op.second << '\n' << e.what() << endl;
      // Debugging
      for(auto [op, phys_reg] : register_allocator->reg_alloc_result_){
        auto [op_name, op_index] = op;
        cout << op_name << '\t' << op_index << ":\t" << phys_reg.index << '\n';
      }

      throw runtime_error("Fatal error. Terminate.\n");
    }
    first |= (reg_idx & 0xff) << 16;
  }

  void SM7X::setSm7xSrcOpEncoding(uint64_t& first, uint64_t& second, Operand const* src, 
    RegisterAllocator const* register_allocator, int loc) const {
    if(src->type_ == IIMM){
      int int_data = src->iimm_;
      if(src->is_neg_) int_data *= -1;
      first |= (*reinterpret_cast<uint64_t*>(&int_data)) << 32;
    } else if(src->type_ == FIMM){
      if(src->data_type_ == F64){
        double double_data = src->fimm_;
        if(src->is_neg_) double_data *= -1.0f;
        first |= (*reinterpret_cast<uint64_t*>(&double_data)) & 0xffffffff00000000;
      } else {
        float float_data = src->fimm_;
        if(src->is_neg_) float_data *= -1.0f;
        first |= (*reinterpret_cast<uint64_t*>(&float_data)) << 32;
      }
    } else if(src->type_ == ID){
      if(src->state_space_ == REG){
        // TODO: is_neg_?
        if(src->is_neg_){
          if(loc == 0){
            second |= 1 << 8;
          }
        }
        uint64_t reg_idx = 0;
        try {reg_idx = register_allocator->reg_alloc_result_.at(src->getOpPair()).index;}
        catch(exception const& e){
          cout << "In setSm7xSrcOpEncoding(), src not found in register allocator.\n"
               << src->name_ << '\n' << e.what() << endl;
          throw runtime_error("Fatal error. Terminate.\n");
        }
        if(loc == 0){
          first |= reg_idx << 24;
        } else if(loc == 1){
          first |= reg_idx << 32;
        } else if(loc == 2){
          second |= reg_idx;
        }
      } else if(src->state_space_ == CREG){
        // must be rz.
        uint64_t rz = 0xff;
        if(loc == 0){
          first |= rz << 24;
        } else if(loc == 1){
          first |= rz << 32;
        } else if(loc == 2){
          second |= rz;
        }
      } else if(src->state_space_ == CONST){
        // predefined_const_bank1 (blockDim_x, blockDim_y, blockDim_z)
        uint64_t bank0 = 0;
        uint64_t bank1 = 0;
        if(predfined_const_bank1.find(src->name_) != predfined_const_bank1.end()){
          bank0 = 0;
          bank1 = predfined_const_bank1.at(src->name_);
        } else {
          bank1 = (uint64_t)(src->param_offset_) + parameter_base_offset;
        }
        first |= bank0 << 54;
        first |= bank1 << 38;
      } else if(src->state_space_ == SREG){
        second |= sreg_encoding.at(src->name_) << 8;
      }
    } else if(src->type_ == MEM_REF){
      if(src->state_space_ == REG){
        uint64_t reg_idx = register_allocator->reg_alloc_result_.at(src->getOpPair()).index;
        if(loc == 0){
          first |= reg_idx << 24;
        } else if(loc == 1){
          first |= reg_idx << 32;
        } else if(loc == 2){
          second |= reg_idx;
        }
      } else if(src->state_space_ == CREG){
        // must be rz.
        uint64_t rz = 0xff;
        if(loc == 0){
          first |= rz << 24;
        } else if(loc == 1){
          first |= rz << 32;
        } else if(loc == 2){
          second |= rz;
        }
      } else if(src->state_space_ == CONST){
        // const mem_ref
      }
      // mem_offset_
      first |= ((uint64_t)src->mem_offset_ & 0xffffff) << 40;
    } else if(src->type_ == LABEL){
      int int_offset = src->label_offset_ * 0x10; // sizeof instr on Sm7x
      first |= (*reinterpret_cast<uint64_t*>(&int_offset)) << 32;
      if(int_offset < 0)
        second |= 0x3ffffULL;
    }
  }

  void SM7X::setSm7xDstPredEncoding(uint64_t& second, Operand const* dst, 
    RegisterAllocator const* register_allocator, int loc) const{
    uint64_t dst_encoding = 0ULL;
    if(dst->state_space_ == CREG){
      if(loc == 0) dst_encoding = 0x7ULL << 17;
      else if(loc == 1) dst_encoding = 0x7ULL << 20;
    } else {
      dst_encoding = register_allocator->preg_alloc_result_.at(dst->getOpPair());
      if(loc == 0) dst_encoding = dst_encoding << 17;
      else if(loc == 1) dst_encoding = dst_encoding << 20;
    }
    second |= dst_encoding;
  }

  void SM7X::setSm7xSrcPredEncoding(uint64_t& second, Operand const* src, 
    RegisterAllocator const* register_allocator, int loc) const{
    uint64_t src_encoding = src->is_neg_? 0x8ULL : 0x0ULL;
    if(src->state_space_ == CREG){
      src_encoding |= 0x7ULL;
      if(loc == 0) src_encoding = src_encoding << 23;
    } else {
      src_encoding |= (uint64_t)register_allocator->preg_alloc_result_.at(src->getOpPair());
      if(loc == 0) src_encoding = src_encoding << 23;
    }
    second |= src_encoding;
  }

  void SM7X::setSm7xSrcPredEncodingDefault(uint64_t& second, int src_loc) const{
    uint64_t src_encoding = 0; // !pt is the default value
    if(src_loc == 0) src_encoding = 0xfULL << 23;
    second |= src_encoding;
  }

  void SM7X::setSm7xDstPredEncodingDefault(uint64_t& second, int dst_loc) const{
    uint64_t dst_encoding = 0x7ULL;
    if(dst_loc == 0) dst_encoding = 0x7ULL << 17;
    else if(dst_loc == 1) dst_encoding = 0x7ULL << 20;
    second |= dst_encoding;
  }

  void SM7X::setSm7xIIMDEncoding(uint64_t& second, Operand const* src, int offset, int width) const {
    int encoding = src->iimm_;
    encoding &= (1ULL << width) - 1;
    second |= encoding << offset;
  }
}