#include <vector>
#include <iostream>

#include "arch_sm70.h"

using namespace std;

namespace dada {
  bool SM70::isVariableLatency(Instruction const* instr){
    return variable_latency.find(instr->opcode_) != variable_latency.end(); 
  }

  int SM70::getLatency(Instruction const* instr){
    int lat = latency.at(instr->opcode_);

    // for special flags. (e.g., IMAD.WIDE)
    if(special_latency.find(instr->opcode_) != special_latency.end()){
      auto const& slat_flags = special_latency.at(instr->opcode_);
      for(auto flag : instr->flags_){
        if(slat_flags.find(flag) != slat_flags.end()){
          lat = slat_flags.at(flag);
          break;
        }
      }
    }
    return lat;
  }

  int SM70::getTputLatency(Instruction const* instr1, Instruction const* instr2) {
    if(instr2->opcode_ == BRA) return 2;
    // if(resource_type.at(instr1->opcode_) == resource_type.at(instr2->opcode_)){
    //   return op_tput.at(instr1->opcode_);
    // } else {
      return 1;
    // }
  }

  int SM70::getParameterBaseOffset() const {
    return parameter_base_offset;
  }

  vector<uint64_t> SM70::getInstructionBinary(Instruction const* instr, RegisterAllocator const* register_allocator, CtrlLogic const& ctrl_logic) const{
    uint64_t first = getOpEncoding(instr);
    uint64_t second = 0;

    first |= getSm7xPredMask(instr, register_allocator);
    second |= getFlagsEncoding(instr);
    second |= getSm7xCtrlLogicEncoding(ctrl_logic);

    // Operands
    // Assume inputs are valid.
    auto opcode = instr->opcode_;
    switch(opcode){
      // All have one operand as dst and fixed number of src
      case HADD2: case HFMA2: case HMUL2: 
      case FADD: case FMUL: case MUFU: 
      case DADD: case DMUL:
      case HMMA: 
      case SHFL:
      case IMNMX: case FMNMX:
      case I2I: case I2F: case F2I: case F2F:
      case SHF: 
      case S2R: case CS2R:
        try{
          setSm7xDstOpEncoding(first, instr->dst_operands_[0].get(), register_allocator);
          for(int src_loc = 0; src_loc < instr->src_operands_.size(); ++src_loc){
            auto const& src = instr->src_operands_[src_loc];
            setSm7xSrcOpEncoding(first, second, src.get(), register_allocator, src_loc);
            modifyICR(opcode, first, src.get(), src_loc); // When src is immed/const, the encoding needs modifying
          }
        } catch(exception const& e){
          cout << "Error when gen operand encoding for hadd2...mov...(SM70)\n"
               << e.what() << endl;
        }
        break;
      case MOV: 
        {
          setSm7xDstOpEncoding(first, instr->dst_operands_[0].get(), register_allocator);
          auto const& src = instr->src_operands_[0];
          setSm7xSrcOpEncoding(first, second, src.get(), register_allocator, 1);
          modifyICR(opcode, first, src.get(), 1);
        }
        break;
      case LDG: case STG: case LDS: case STS: case LDC:
        if(instr->dst_operands_.size() != 0)
          setSm7xDstOpEncoding(first, instr->dst_operands_[0].get(), register_allocator);
        for(int src_loc = 0; src_loc < instr->src_operands_.size(); ++src_loc){
          auto const& src = instr->src_operands_[src_loc];
          setSm7xSrcOpEncoding(first, second, src.get(), register_allocator, src_loc);
          modifyICR(opcode, first, src.get(), src_loc); // When src is immed/const, the encoding needs modifying
        }
        break;
      case IADD3: 
        {
          setSm7xDstOpEncoding(first, instr->dst_operands_[0].get(), register_allocator);
          if(instr->dst_operands_.size() == 1){
            setSm7xDstPredEncodingDefault(second, 0);
            setSm7xDstPredEncodingDefault(second, 1);
          } else if(instr->dst_operands_.size() == 3){
            setSm7xDstPredEncoding(second, instr->dst_operands_[1].get(), register_allocator, 0);
            setSm7xDstPredEncoding(second, instr->dst_operands_[2].get(), register_allocator, 1);
          }

          for(int src_loc = 0; src_loc < 3; ++src_loc){
            auto const& src = instr->src_operands_[src_loc];
            setSm7xSrcOpEncoding(first, second, src.get(), register_allocator, src_loc);
            modifyICR(opcode, first, src.get(), src_loc); // When src is immed/const, the encoding needs modifying
          }

          bool has_carryin = false;
          if(instr->flags_.size() == 1 && instr->flags_[0] == X_FLAG) has_carryin = true;
          if(has_carryin){
            setSm7xSrcPredEncoding(second, instr->src_operands_[3].get(), register_allocator, 0);
            setSm7xSrcPredEncoding(second, instr->src_operands_[4].get(), register_allocator, 1);
          }
        }
        break;
      case FFMA: case DFMA: 
        {
          setSm7xDstOpEncoding(first, instr->dst_operands_[0].get(), register_allocator);
          bool has_ic_suffix = false;
          if(instr->src_operands_[2]->state_space_ == CONST ||
             instr->src_operands_[2]->type_ == IIMM || 
             instr->src_operands_[2]->type_ == FIMM){ has_ic_suffix = true;}
          for(int src_loc = 0; src_loc < instr->src_operands_.size(); ++src_loc){
            auto const& src = instr->src_operands_[src_loc];
            int src_loc_offset = (src_loc == 1 && has_ic_suffix)? 1 : 0;
            setSm7xSrcOpEncoding(first, second, src.get(), register_allocator, src_loc+src_loc_offset);
            modifyICR(opcode, first, src.get(), src_loc); // When src is immed/const, the encoding needs modifying
          }
        }
        break;
      case IMAD:
        {
          setSm7xDstOpEncoding(first, instr->dst_operands_[0].get(), register_allocator);

          bool has_ic_suffix = false;
          if(instr->src_operands_[2]->state_space_ == CONST ||
             instr->src_operands_[2]->type_ == IIMM){ has_ic_suffix = true;}
             
          for(int src_loc = 0; src_loc < instr->src_operands_.size(); ++src_loc){
            auto const& src = instr->src_operands_[src_loc];
            int src_loc_offset = (src_loc == 1 && has_ic_suffix)? 1 : 0;
            setSm7xSrcOpEncoding(first, second, src.get(), register_allocator, src_loc+src_loc_offset);
            modifyICR(opcode, first, src.get(), src_loc); // When src is immed/const, the encoding needs modifying
          }
          setSm7xDstPredEncodingDefault(second, 0);
        }
        break;
      case LEA:
        {
          setSm7xDstOpEncoding(first, instr->dst_operands_[0].get(), register_allocator);
          if(instr->dst_operands_.size() == 2){
            setSm7xDstPredEncoding(second, instr->dst_operands_[1].get(), register_allocator, 0);
          } else {
            setSm7xDstPredEncodingDefault(second, 0);
          }

          bool has_carryin = false;

          for(int src_loc = 0; src_loc < 2; ++src_loc){
            auto const& src = instr->src_operands_[src_loc];
            setSm7xSrcOpEncoding(first, second, src.get(), register_allocator, src_loc);
            modifyICR(opcode, first, src.get(), src_loc); // When src is immed/const, the encoding needs modifying
          }
          setSm7xIIMDEncoding(second, instr->src_operands_[2].get(), 11, 5);

          for(auto const& flag : instr->flags_){
            if(flag == X_FLAG){
              has_carryin = true;
              break;
            }
          }
          if(has_carryin){
            setSm7xSrcPredEncoding(second, instr->src_operands_[3].get(), register_allocator, 0);
          }
        }
        break;
      case ISETP: case PSETP:
        {
          setSm7xDstPredEncoding(second, instr->dst_operands_[0].get(), register_allocator, 0);
          setSm7xDstPredEncoding(second, instr->dst_operands_[1].get(), register_allocator, 1);
          setSm7xSrcOpEncoding(first, second, instr->src_operands_[0].get(), register_allocator, 0);
          setSm7xSrcOpEncoding(first, second, instr->src_operands_[1].get(), register_allocator, 1);
          modifyICR(instr->opcode_, first, instr->src_operands_[1].get(), 1);
          setSm7xSrcPredEncoding(second, instr->src_operands_[2].get(), register_allocator, 0);
        }
        break;
      case LOP3:
        {
          setSm7xDstOpEncoding(first, instr->dst_operands_[0].get(), register_allocator);
          setSm7xDstPredEncodingDefault(second, 0);
          setSm7xSrcPredEncodingDefault(second, 0);


          for(int src_loc = 0; src_loc < 3; ++src_loc){
            auto const& src = instr->src_operands_[src_loc];
            setSm7xSrcOpEncoding(first, second, src.get(), register_allocator, src_loc);
            modifyICR(opcode, first, src.get(), src_loc); // When src is immed/const, the encoding needs modifying
          }

          setSm7xIIMDEncoding(second, instr->src_operands_[3].get(), 8, 8);
        }
        break;
      case P2R: case R2P:
        break;      
      case BRA:
        {
          setSm7xSrcOpEncoding(first, second, instr->src_operands_[0].get(), register_allocator, 1);
        }
      case EXIT: case NOP:
        break; // No operand
      case IMMA: case BMMA: 
        throw runtime_error("sm70 devices do not support IMMA/BMMA.\n");
        break;
    }

    // Epilogue (try to merge them into flags/operands)
    switch(instr->opcode_){
      case LDC:
        first |= 0xff000000;
        break;
      case LDG: case STG:
        second |= 0x7 << 17;
        second |= 0x1 << 8; // .E (64-bit)
        break;
      case MOV: 
        second |= 0xf00;
        break;
      case EXIT:
        second |= 0x7 << 23;
        break;
      case ISETP:
        second |= 0x7 << 4;
        second |= 0x1 << 9; // .s32 (default)
        second &= ~(0x1ULL << 45);
        break;
      case SHF:
        second |= 0x6<<8; // .s32
        break;
      case BRA:
        second |= 0x7<<23;
        // first |= 0x1ULL<<32; // .u
        break;
      // case LDS:
      //   second |= 1<<12; // .u
      //   break;
    }

    
    vector<uint64_t> binary_data {first, second};
    return binary_data;
  }

//--------------------------Helper-functions-------------------------
  uint64_t SM70::getOpEncoding(Instruction const* instr) const {
    auto const& opcode = instr->opcode_;
    if(alter_op_encoding_.find(opcode) != alter_op_encoding_.end()){
      auto const& alter_flags = alter_op_encoding_.at(opcode);
      for(auto const& flag : instr->flags_){
        if(alter_flags.find(flag) != alter_flags.end()){
          return alter_flags.at(flag);
        }
      }
    }    
    return op_encoding_.at(opcode);
  }

  uint64_t SM70::getFlagsEncoding(Instruction const* instr) const {
    uint64_t result = 0;
    for(auto const& flag : instr->flags_){
      result |= flag_encoding_.at(flag);
    }
    return result;
  }

  void SM70::modifyICR(Opcode opcode, uint64_t& first, Operand const* src, int loc) const {
    // some ops have different behavior.
    if(opcode == FADD && loc == 1) loc = 2;

    if(src->type_ == FIMM || src->type_ == IIMM){
      if(loc == 1){
        first &= ~(1ULL << 9);
        first |= 0x8ULL << 8;
      } else if(loc == 2){
        first &= ~(1ULL << 9);
        first |= 0x4ULL << 8;
      }
    } else if(src->state_space_ == CONST){
      if(loc == 1){
        first &= ~(1ULL << 9);
        first |= 0xaULL << 8;
      } else if(loc == 2){
        first &= ~(1ULL << 9);
        first |= 0x6ULL << 8;
      }
    }
  }
}