#include <iostream>
#include "instruction.h"

using namespace std;

namespace dada {
  Instruction::Instruction(Opcode opcode, vector<Flag> flags, vector<Operand*> operands) : opcode_(opcode), flags_(flags) {
    for(auto& op : operands){
      operands_.emplace_back(op);
    }
  }

  Instruction::Instruction(Operand* pred_mask, Opcode opcode, vector<Flag> flags, vector<Operand*> operands) : opcode_(opcode), flags_(flags) {
    predicate_mask_.reset(pred_mask);
    for(auto& op : operands){
      operands_.emplace_back(op);
    }
  }

  void Instruction::checkSemantic() {
    map<Flag, int> const ldst_width = {
      {LDST16, 16}, {LDST32, 32}, {LDST64, 64}, {LDST128, 128},
    };
    switch(opcode_){
      case HADD2:
        {
          if(flags_.size() != 0) throw runtime_error("HADD2 accepts no flag.\n");
          if(operands_.size() != 3) throw runtime_error("HADD2 can only accept 2 source operands.\n");
          auto& dst = operands_[0];
          auto& src0 = operands_[1];
          auto& src1 = operands_[2];
          if(dst->type_ != ID || dst->state_space_ != REG || dst->data_type_ != F16X2)
            throw runtime_error("HADD2 has invalid dst.\n");
          if(src0->type_ != ID || src0->state_space_ != REG || src0->data_type_ != F16X2)
            throw runtime_error("HADD2's first src is invalid.\n");
          if(src1->type_ != ID || src1->state_space_ != REG || src1->data_type_ != F16X2)
            throw runtime_error("HADD2's second src is invalid.\n");
          

          dst_operands_.emplace_back(dst.release());
          src_operands_.emplace_back(src0.release());
          src_operands_.emplace_back(src1.release());
        }
        break;
      case HMUL2:
        checkFloatFlags();
        throw runtime_error("HMUL2 is not supported yet.\n");
        break;
      case HFMA2:
        {
          if(flags_.size() != 0) throw runtime_error("HFMA2 accepts no flag.\n");
          if(operands_.size() != 4) throw runtime_error("HFMA2 can only accept 3 source operands.\n");
          auto& dst = operands_[0];
          auto& src0 = operands_[1];
          auto& src1 = operands_[2];
          auto& src2 = operands_[3];
          if(dst->type_ != ID || dst->state_space_ != REG || dst->data_type_ != F16X2)
            throw runtime_error("HADD2 has invalid dst.\n");
          if(src0->type_ != ID || src0->state_space_ != REG || src0->data_type_ != F16X2)
            throw runtime_error("HADD2's first src is invalid.\n");
          if(src1->type_ != ID || src1->state_space_ != REG || src1->data_type_ != F16X2)
            throw runtime_error("HADD2's second src is invalid.\n");
          if(src2->type_ != ID || src2->state_space_ != REG || src2->data_type_ != F16X2)
            throw runtime_error("HADD2's second src is invalid.\n");
          

          dst_operands_.emplace_back(dst.release());
          src_operands_.emplace_back(src0.release());
          src_operands_.emplace_back(src1.release());
          src_operands_.emplace_back(src2.release());
        }
        break;

      case FADD:
        checkFloatFlags();
        {
          if(operands_.size() != 3) throw runtime_error("FADD can only accept 2 source operands.\n");
          auto& dst = operands_[0];
          auto& src0 = operands_[1];
          auto& src1 = operands_[2];
          if(dst->type_ != ID || dst->state_space_ != REG || dst->data_type_ != F32)
            throw runtime_error("FADD has invalid dst.\n");
          if(src0->type_ != ID || src0->state_space_ != REG || src0->data_type_ != F32)
            throw runtime_error("FADD's first src is invalid.\n");
          if(src1->type_ != FIMM){
            if(src1->type_ != ID || src1->data_type_ != F32)
              throw runtime_error("FADD's second src is invalid.\n");
          }

          dst_operands_.emplace_back(dst.release());
          src_operands_.emplace_back(src0.release());
          src_operands_.emplace_back(src1.release());
        }
        break;
      case FMUL:
        checkFloatFlags();
        {
          if(operands_.size() != 3) throw runtime_error("FMUL can only accept 2 source operands.\n");
          auto& dst = operands_[0];
          auto& src0 = operands_[1];
          auto& src1 = operands_[2];
          if(dst->type_ != ID || dst->state_space_ != REG || dst->data_type_ != F32)
            throw runtime_error("FMUL has invalid dst.\n");
          if(src0->type_ != ID || src0->state_space_ != REG || src0->data_type_ != F32)
            throw runtime_error("FMUL's first src is invalid.\n");
          if(src1->type_ != FIMM){
            if(src1->type_ != ID || src1->data_type_ != F32)
              throw runtime_error("FMUL's second src is invalid.\n");
          }

          dst_operands_.emplace_back(dst.release());
          src_operands_.emplace_back(src0.release());
          src_operands_.emplace_back(src1.release());
        }
        break;
      case FFMA:
        checkFloatFlags();
        {
          if(operands_.size() != 4) throw runtime_error("FFMA can only accept 3 source operands.\n");
          auto& dst = operands_[0];
          auto& src0 = operands_[1];
          auto& src1 = operands_[2];
          auto& src2 = operands_[3];
          if(dst->type_ != ID || dst->state_space_ != REG || dst->data_type_ != F32)
            throw runtime_error("FFMA has invalid dst.\n");
          if(src0->type_ != ID || src0->state_space_ != REG || src0->data_type_ != F32)
            throw runtime_error("FFMA's first src is invalid.\n");
          if(src1->type_ != FIMM){
            if(src1->type_ != ID || src1->data_type_ != F32)
              throw runtime_error("FFMA's second src is invalid.\n");
          }

          dst_operands_.emplace_back(dst.release());
          src_operands_.emplace_back(src0.release());
          src_operands_.emplace_back(src1.release());
          src_operands_.emplace_back(src2.release());
        }
        break;
      case MUFU:
        break;

      case DADD:
        checkFloatFlags();
        {
          if(operands_.size() != 3) throw runtime_error("DADD can only accept 2 source operands.\n");
          auto& dst = operands_[0];
          auto& src0 = operands_[1];
          auto& src1 = operands_[2];
          if(dst->type_ != ID || dst->state_space_ != REG || dst->data_type_ != F64)
            throw runtime_error("FADD has invalid dst.\n");
          if(src0->type_ != ID || src0->state_space_ != REG || src0->data_type_ != F64)
            throw runtime_error("FADD's first src is invalid.\n");
          if(src1->type_ != ID || src1->state_space_ != REG || src1->data_type_ != F64)
            throw runtime_error("FADD's second src is invalid.\n");

          dst_operands_.emplace_back(dst.release());
          src_operands_.emplace_back(src0.release());
          src_operands_.emplace_back(src1.release());
        }
        break;
      case DMUL:
        checkFloatFlags();
        break;
      case DFMA:
        checkFloatFlags();
        break;

      case HMMA:
        {
          if(flags_.size() != 2) throw runtime_error("HMMA has invalid flags.\n");
          if(flags_[0] != HMMA1688 || (flags_[1] != F16T && flags_[1] != F32T))
            throw runtime_error("HMMA has invalid flags.\n");

          if(operands_.size() != 4) throw runtime_error("HMMA excepts 4 operands.\n");
          auto& dst = operands_[0];
          auto& src0 = operands_[1];
          auto& src1 = operands_[2];
          auto& src2 = operands_[3];

          int accumualator_size = flags_[1] == F16T? 64 : 128;

          if(dst->type_ != ID || dst->state_space_ != REG || dst->getBitWidth() != accumualator_size)
            throw runtime_error("HMMA's dst is invalid.\n");
          if(src0->type_ != ID || src0->state_space_ != REG || src0->getBitWidth() != 64)
            throw runtime_error("HMMA's src0 is invalid.\n");
          if(src1->type_ != ID || src1->state_space_ != REG || src1->getBitWidth() != 32)
            throw runtime_error("HMMA's src1 is invalid.\n");
          if(src2->type_ != ID || src2->state_space_ != REG || src2->getBitWidth() != accumualator_size)
            throw runtime_error("HMMA's src2 is invalid.\n");

          dst_operands_.emplace_back(dst.release());
          src_operands_.emplace_back(src0.release());
          src_operands_.emplace_back(src1.release());
          src_operands_.emplace_back(src2.release());          
        }
        break;
      case IMMA:
        break;
      case BMMA:
        break;

      case LDG:
        {
          checkGmemFlags();
          int width = getLdstWidth();
          
          // ldg dst, [ptr];
          if(operands_.size() != 2) throw runtime_error("LDG can only accept 2 oreands.\n");
          if(operands_[0]->getBitWidth() != width) 
            throw runtime_error("LDG's flag indicate width" + to_string(width) + 
                       ", while operand width is " + to_string(operands_[0]->getBitWidth()));
          // if(operands_[1]->getBitWidth() != 64) throw runtime_error("LDG's width mush be 64-bit wide.\n");
          if(operands_[0]->type_ != ID      || operands_[0]->state_space_ != REG || 
             operands_[1]->type_ != MEM_REF || (operands_[1]->state_space_ != REG  && operands_[1]->state_space_ != CREG))
               throw runtime_error("LDG has invalid opereands.\n");

          dst_operands_.emplace_back(operands_[0].release());
          src_operands_.emplace_back(operands_[1].release());
        }
        break;
      case STG:
        {
          checkGmemFlags();
          int width = getLdstWidth();
          
          // stg [ptr], src;
          if(operands_.size() != 2) throw runtime_error("STG can only accept 2 operands.\n");
          if(operands_[1]->getBitWidth() != width) throw runtime_error("STG width mismatch.\n");
          if(operands_[1]->type_ != ID      || operands_[1]->state_space_ != REG || 
             operands_[0]->type_ != MEM_REF || (operands_[0]->state_space_ != REG  && operands_[0]->state_space_ != CREG))
               throw runtime_error("STG has invalid opereands.\n");

          src_operands_.emplace_back(operands_[0].release());
          src_operands_.emplace_back(operands_[1].release());
        }
        break;
      case LDS:
        {
          checkSmemFlags();
          int width = getLdstWidth();

          if(operands_.size() != 2) throw runtime_error("LDS excepts 2 operands.\n");
          auto& dst = operands_[0];
          auto& src = operands_[1];
          if(dst->getBitWidth() != width || dst->state_space_ != REG)
            throw runtime_error("LDS' dst is invalid.\n");
          if(src->type_ != MEM_REF || 
             !(src->name_ == "rz" || (src->state_space_ == REG && src->data_type_ == S32)) )
            throw runtime_error("LDS' src is invalid.\n");

          dst_operands_.emplace_back(dst.release());
          src_operands_.emplace_back(src.release());
        }
        break;
      case STS:
        {
          checkSmemFlags();
          int width = getLdstWidth();

          if(operands_.size() != 2) throw runtime_error("STS excepts 2 operands.\n");
          auto& src0 = operands_[0];
          auto& src1 = operands_[1];

          if(src0->type_ != MEM_REF || src0->data_type_ != S32)
            throw runtime_error("STS' src0 is invalid.\n");
          if(src1->name_ != "rz" && src1->getBitWidth() != width)
            throw runtime_error("STS' src1 is invalid.\n");
          
          src_operands_.emplace_back(src0.release());
          src_operands_.emplace_back(src1.release());
        }
        break;
      case LDC:
        {
          if(flags_.size() == 0 || 
            (flags_.size() == 1 && ldst_width.find(flags_[0]) != ldst_width.end())){
            int width = getLdstWidth();
            if(operands_.size() == 2 && 
              (operands_[0]->state_space_ == REG && operands_[1]->state_space_ == CONST) &&
              (operands_[0]->getBitWidth() == width && operands_[1]->getBitWidth() == width)){
              dst_operands_.emplace_back(operands_[0].release());
              src_operands_.emplace_back(operands_[1].release());
            } else {
              throw runtime_error("LDC has invalid operands.\n");
            }
          } else {
            throw runtime_error("LDC encounters invalid flags.\n");
          }
        }
        break;

      case MOV:
        {
          if(flags_.size() != 0) throw runtime_error("MOV encounters invalid flags.\n");
          if(operands_.size() != 2) throw runtime_error("MOV can only accept 2 operands.\n");
          auto& dst = operands_[0];
          auto& src = operands_[1];

          if(dst->state_space_ != REG || dst->getBitWidth() != 32)
            throw runtime_error("MOV's dst is invalid.\n");
          if(src->state_space_ == SREG || 
            (src->type_ == ID && src->state_space_ == REG && src->getBitWidth() != 32))
            throw runtime_error("MOV's src is invalid.\n");

          dst_operands_.emplace_back(dst.release());
          src_operands_.emplace_back(src.release());          
        }
        break;
      case SHFL:
        break;

      case IADD3:
        {
          if(flags_.size() > 1) throw runtime_error("IADD3 accepts no more than one flag.\n");
          if(flags_.size() == 1 && flags_[0] != X_FLAG) throw runtime_error("IADD3 accepts .x only.\n");
          bool is_x = false;
          if(flags_.size() == 1 && flags_[0] == X_FLAG) is_x = true;

          if(operands_.size() != 4 && operands_.size() != 6)
            throw runtime_error("IADD3 has invalid operands.\n");
          if(is_x && operands_.size() != 6)
            throw runtime_error("IADD3 has invalid operands.\n");

          auto& dst = operands_[0];
          if(dst->state_space_ != REG || dst->data_type_ != S32)
              throw runtime_error("IADD3 has invalid dst.\n");
          dst_operands_.emplace_back(dst.release());

          if(operands_.size() == 6){
            if(!is_x){
              auto& dst_p0 = operands_[1];
              auto& dst_p1 = operands_[2];
              if(dst_p0->data_type_ != PRED || dst_p1->data_type_ != PRED)
                throw runtime_error("IADD's dst1&2 are expected to be predicates.\n");
              dst_operands_.emplace_back(dst_p0.release());
              dst_operands_.emplace_back(dst_p1.release());

              auto& src0 = operands_[3];
              auto& src1 = operands_[4];
              auto& src2 = operands_[5];            

              if(!(src0->name_ == "rz" || 
                  (src0->state_space_ == REG && src0->data_type_ == S32) ))
                throw runtime_error("IADD3's src0 is invalid.\n");
              if(!(src1->name_ == "rz" || 
                  (src1->state_space_ == REG && src1->data_type_ == S32) || 
                  (src1->state_space_ == CONST) || 
                  (src1->type_ == IIMM) ))
                throw runtime_error("IADD3's src1 is invalid.\n");
              if(!(src2->name_ == "rz" ||
                  (src2->state_space_ == REG && src2->data_type_ == S32) ||
                  (src2->state_space_ == CONST) || 
                  (src2->type_ == IIMM)))
                throw runtime_error("IADD3's src2 is invalid.\n");

              src_operands_.emplace_back(src0.release());
              src_operands_.emplace_back(src1.release());
              src_operands_.emplace_back(src2.release());
            } else { // iadd3.x
              auto& src0 = operands_[1];
              auto& src1 = operands_[2];
              auto& src2 = operands_[3];
              auto& src_p0 = operands_[4];
              auto& src_p1 = operands_[5];

              if(!(src0->name_ == "rz" || 
                  (src0->state_space_ == REG && src0->data_type_ == S32) ))
                throw runtime_error("IADD3's src0 is invalid.\n");
              if(!(src1->name_ == "rz" || 
                  (src1->state_space_ == REG && src1->data_type_ == S32) || 
                  (src1->state_space_ == CONST) || 
                  (src1->type_ == IIMM) ))
                throw runtime_error("IADD3's src1 is invalid.\n");
              if(!(src2->name_ == "rz" ||
                  (src2->state_space_ == REG && src2->data_type_ == S32) ||
                  (src2->state_space_ == CONST) || 
                  (src2->type_ == IIMM)))
                throw runtime_error("IADD3's src2 is invalid.\n");

              if(src_p0->data_type_ != PRED || src_p1->data_type_ != PRED)
                throw runtime_error("IADD.x's src_p0&src_p1 are expected to be predicates.\n");

              src_operands_.emplace_back(src0.release());
              src_operands_.emplace_back(src1.release());
              src_operands_.emplace_back(src2.release());
              src_operands_.emplace_back(src_p0.release());
              src_operands_.emplace_back(src_p1.release());
            }

          } else if(operands_.size() == 4) {
            auto& src0 = operands_[1];
            auto& src1 = operands_[2];
            auto& src2 = operands_[3];

            if(!(src0->name_ == "rz" || 
                (src0->state_space_ == REG && src0->data_type_ == S32) ))
              throw runtime_error("IADD3's src0 is invalid.\n");
            if(!(src1->name_ == "rz" || 
                (src1->state_space_ == REG && src1->data_type_ == S32) || 
                (src1->state_space_ == CONST) || 
                (src1->type_ == IIMM) ))
              throw runtime_error("IADD3's src1 is invalid.\n");
            if(!(src2->name_ == "rz" ||
                (src2->state_space_ == REG && src2->data_type_ == S32) ||
                (src2->state_space_ == CONST) || 
                (src2->type_ == IIMM)))
              throw runtime_error("IADD3's src2 is invalid.\n");

            src_operands_.emplace_back(src0.release());
            src_operands_.emplace_back(src1.release());
            src_operands_.emplace_back(src2.release());
          }


        }
        break;
      case IMAD:
        {
          if(flags_.size() > 1) throw runtime_error("IMAD only supports .wide flag for now.\n");
          bool is_wide = false;
          if(flags_.size() == 1)
            if(flags_[0] != WIDE) throw runtime_error("IMAD only supports .wide flag for now.\n");
            else is_wide = true;
          flags_.push_back(IS32);

          if(operands_.size() != 4) throw runtime_error("IMAD excepts 4 operands.\n");
          auto& dst = operands_[0];
          auto& src0 = operands_[1];
          auto& src1 = operands_[2];
          auto& src2 = operands_[3];

          if(is_wide){
            if(dst->data_type_ != S64 || dst->state_space_ != REG)
              throw runtime_error("IMAD's dst is invalid.\n");
            if(src2->data_type_ != S64)
              throw runtime_error("IMAD's src2 has invalid data type.\n");
          } else {
            if(dst->data_type_ != S32 || dst->state_space_ != REG)
              throw runtime_error("IMAD's dst is invalid.\n");
            if(!(src2->name_ == "rz" || src2->data_type_ == S32))
              throw runtime_error("IMAD's src2 has invalid data type.\n");
          }
          if(src0->state_space_ == REG)
            if(src0->data_type_ != S32 && src0->data_type_ != U32)
              throw runtime_error("IMAD's src0 is invalid.\n");
          if(src0->state_space_ == SREG)
            throw runtime_error("IMAD's src0 cannot be SREG.\n");
          if(!(src1->name_ == "rz" || 
              (src1->type_ == ID && src1->state_space_ == REG && src1->data_type_ == S32) ||
              (src1->type_ == IIMM) ||
              (src1->state_space_ == CONST) ))
            throw runtime_error("IMAD's src1 is invalid.\n");
          if(src1->state_space_ == SREG)
            throw runtime_error("IMAD's src1 cannot be SREG.\n");

          dst_operands_.emplace_back(dst.release());
          src_operands_.emplace_back(src0.release());
          src_operands_.emplace_back(src1.release());
          src_operands_.emplace_back(src2.release());
        }
        break;
      case LEA:
        {
          bool is_x = false;
          if(flags_.size() > 1) throw runtime_error("LEA excepts <= 1 flags.\n");
          if(flags_.size() == 1){
            if(flags_[0] == X_FLAG) is_x = true;
            else throw runtime_error("LEA can only accept .X as flag.\n");
          }

          auto& dst = operands_[0];
          if(dst->state_space_ != REG || dst->data_type_ != S32)
                throw runtime_error("LEA has invalid dst.\n");
          dst_operands_.emplace_back(dst.release());

          if(!is_x){
            if(!(operands_.size() == 4 || operands_.size() == 5)) 
              throw runtime_error("LEA accepts 4/5 operands.\n");
            if(operands_.size() == 4){ // lea a, b, c, 0x2;              
              auto& src0 = operands_[1];
              auto& src1 = operands_[2];
              auto& src2 = operands_[3];
              
              if(!(src0->name_ == "rz" || 
                  (src0->state_space_ == REG && src0->data_type_ == S32) ))
                throw runtime_error("LEA's src0 is invalid.\n");
              if(!(src1->name_ == "rz" || 
                  (src1->state_space_ == REG && src1->data_type_ == S32) || 
                  (src1->state_space_ == CONST) || 
                  (src1->type_ == IIMM) ))
                throw runtime_error("LEA's src1 is invalid.\n");
              if(src2->type_ != IIMM) throw runtime_error("LEA's src2 is invalid.\n");
              if(src2->iimm_ >= 32 || src2->iimm_ < 0)
                throw runtime_error("LEA's src2 must between 0 and 31.\n");
              
              src_operands_.emplace_back(src0.release());
              src_operands_.emplace_back(src1.release());
              src_operands_.emplace_back(src2.release());
            } else if (operands_.size() == 5){
              auto& dst_p = operands_[1];
              auto& src0 = operands_[2];
              auto& src1 = operands_[3];
              auto& src2 = operands_[4];

              if(dst_p->data_type_ != PRED)
                throw runtime_error("LEA's dst1 must be predicate reg.\n");
              
              if(!(src0->name_ == "rz" || 
                  (src0->state_space_ == REG && src0->data_type_ == S32) ))
                throw runtime_error("LEA's src0 is invalid.\n");
              if(!(src1->name_ == "rz" || 
                  (src1->state_space_ == REG && src1->data_type_ == S32) || 
                  (src1->state_space_ == CONST) || 
                  (src1->type_ == IIMM) ))
                throw runtime_error("LEA's src1 is invalid.\n");
              if(src2->type_ != IIMM) throw runtime_error("LEA's src2 is invalid.\n");
              if(src2->iimm_ >= 32 || src2->iimm_ < 0)
                throw runtime_error("LEA's src2 must between 0 and 31.\n");

              dst_operands_.emplace_back(dst_p.release());
              src_operands_.emplace_back(src0.release());
              src_operands_.emplace_back(src1.release());
              src_operands_.emplace_back(src2.release());
            }
          } else { // lea.x dst, base, offset, 0x2, p0;
            if(operands_.size() != 5) throw runtime_error("lea.x excepts 5 operands.\n");
              auto& src0 = operands_[1];
              auto& src1 = operands_[2];
              auto& src2 = operands_[3];
              auto& src_p = operands_[4];

              if(!(src0->name_ == "rz" || 
                  (src0->state_space_ == REG && src0->data_type_ == S32) ))
                throw runtime_error("LEA's src0 is invalid.\n");
              if(!(src1->name_ == "rz" || 
                  (src1->state_space_ == REG && src1->data_type_ == S32) || 
                  (src1->state_space_ == CONST) || 
                  (src1->type_ == IIMM) ))
                throw runtime_error("LEA's src1 is invalid.\n");
              if(src2->type_ != IIMM) throw runtime_error("LEA's src2 is invalid.\n");
              if(src2->iimm_ >= 32 || src2->iimm_ < 0)
                throw runtime_error("LEA's src2 must between 0 and 31.\n");
              if(src_p->data_type_ != PRED)
                throw runtime_error("LEA.x's src3 must be predicate.\n");

              src_operands_.emplace_back(src0.release());
              src_operands_.emplace_back(src1.release());
              src_operands_.emplace_back(src2.release());
              src_operands_.emplace_back(src_p.release());
          }
        }
        break;

      case ISETP:
        {
          if(flags_.size() != 2) throw runtime_error("ISETP excepts 2 flags.\n");
          set<Flag> const icmp_flags = {EQ, NE, LT, GT, GE, LE};
          set<Flag> const bool_flags = {AND, XOR, OR};
          Flag flag0 = flags_[0];
          Flag flag1 = flags_[1];
          if(icmp_flags.find(flag0) == icmp_flags.end() || bool_flags.find(flag1) == bool_flags.end())
            throw runtime_error("ISETP has invalid flag.\n");
          
          // isetp.lt.and p0, pt, tid, n, pt;
          if(operands_.size() != 5) throw runtime_error("ISETP excepts 5 operands.\n");
          auto& dst0 = operands_[0];
          auto& dst1 = operands_[1];
          auto& src0 = operands_[2];
          auto& src1 = operands_[3];
          auto& src2 = operands_[4];
          if(dst0->data_type_ != PRED || dst1->data_type_ != PRED)
            throw runtime_error("ISETP's dsts must be pred.\n");
          if(!(src0->name_ == "rz" || (src0->state_space_ == REG && src0->data_type_ == S32)))
            throw runtime_error("ISETP's src0 is invalid.\n");
          if(!(src1->name_ == "rz" || 
            (src1->state_space_ == REG && src1->data_type_ == S32) || 
            (src1->type_ == IIMM) ||
            (src1->state_space_ == CONST)))
            throw runtime_error("ISETP's src1 is invalid.\n");
          if(src2->data_type_ != PRED)
            throw runtime_error("ISETP's src2 must be pred.\n");
          
          dst_operands_.emplace_back(dst0.release());
          dst_operands_.emplace_back(dst1.release());
          src_operands_.emplace_back(src0.release());
          src_operands_.emplace_back(src1.release());
          src_operands_.emplace_back(src2.release());
        }
        break;
      case PSETP:
        break;

      case SHF:
        {
          if(flags_.size() != 1) throw runtime_error("SHF's flag is invalid.\n");
          if(flags_[0] != SHIFT_L && flags_[0] != SHIFT_R) throw runtime_error("SHF's flag is invalid.\n");
          
          if(operands_.size() != 4) throw runtime_error("SHF expects 4 operands.\n");

          auto& dst =  operands_[0];
          auto& src0 = operands_[1];
          auto& src1 = operands_[2];
          auto& src2 = operands_[3];

          if(dst->state_space_ != REG || dst->data_type_ != S32)
              throw runtime_error("SHF has invalid dst.\n");
          dst_operands_.emplace_back(dst.release());

          if(!(src0->name_ == "rz" || 
              (src0->state_space_ == REG && src0->data_type_ == S32) ))
            throw runtime_error("IADD3's src0 is invalid.\n");
          if(!(src1->name_ == "rz" || 
              (src1->state_space_ == REG && src1->data_type_ == S32) || 
              (src1->state_space_ == CONST) || 
              (src1->type_ == IIMM) ))
            throw runtime_error("IADD3's src1 is invalid.\n");
          if(!(src2->name_ == "rz" ||
              (src2->state_space_ == REG && src2->data_type_ == S32) ||
              (src2->state_space_ == CONST) || 
              (src2->type_ == IIMM)))
            throw runtime_error("IADD3's src2 is invalid.\n");

          src_operands_.emplace_back(src0.release());
          src_operands_.emplace_back(src1.release());
          src_operands_.emplace_back(src2.release());
        }
        break;
      case LOP3:
        {
          if(flags_.size() != 1) throw runtime_error("LOP3's flag is invalid.\n");
          if(flags_[0] != LUT) throw runtime_error("LOP3's flag can only be .lut.\n");

          if(operands_.size() != 5) throw runtime_error("LOP3 expects 5 operands.\n");
          auto& dst = operands_[0];
          auto& src0 = operands_[1];
          auto& src1 = operands_[2];
          auto& src2 = operands_[3];
          auto& src3 = operands_[4];

          if(dst->state_space_ != REG || dst->data_type_ != S32)
            throw runtime_error("LOP3's dst is invalid.\n");
          dst_operands_.emplace_back(dst.release());

          if(!(src0->name_ == "rz" || 
              (src0->state_space_ == REG && src0->data_type_ == S32) ))
            throw runtime_error("LOP3's src0 is invalid.\n");
          if(!(src1->name_ == "rz" || 
              (src1->state_space_ == REG && src1->data_type_ == S32) || 
              (src1->state_space_ == CONST) || 
              (src1->type_ == IIMM) ))
            throw runtime_error("LOP3's src1 is invalid.\n");
          if(!(src2->name_ == "rz" ||
              (src2->state_space_ == REG && src2->data_type_ == S32) ||
              (src2->state_space_ == CONST) || 
              (src2->type_ == IIMM)))
            throw runtime_error("LOP3's src2 is invalid.\n");
          if(src3->type_ != IIMM)
            throw runtime_error("LOP3's src3 is invalid.\n");

          src_operands_.emplace_back(src0.release());
          src_operands_.emplace_back(src1.release());
          src_operands_.emplace_back(src2.release());
          src_operands_.emplace_back(src3.release());
        }
        break;

      case I2I:
        break;
      case I2F:
        break;
      case F2I:
        break;
      case F2F:
        break;

      case BRA:
        {
          if(flags_.size() != 0) throw runtime_error("BRA cannot accept any flag.\n");
          if(operands_.size() != 1) throw runtime_error("BRA can only accept one operand.\n");
          if(operands_[0]->type_ != LABEL) throw runtime_error("BRA can only accept one label.\n");
          src_operands_.emplace_back(operands_[0].release());
        }
        break;
      case EXIT:
        {
          if(operands_.size() != 0) throw runtime_error("EXIT cannot accept any operand.\n");
          if(flags_.size() != 0) throw runtime_error("EXIT cannot accept any flag.\n");
        }
        break;
      case JMP:
        break;

      case P2R:
        break;
      case R2P:
        break;

      case BAR:
        {
        }
        break;
      case CS2R:
        {
          if(flags_.size() > 1) throw runtime_error("CS2R excepts no more than one flag.\n");
          int width = 64; // default
          if(flags_.size() == 1 && !(flags_[0] == LDST32 || flags_[0] == LDST64))
            throw runtime_error("CS2R has invalid flag.\n");
          else if(flags_.size() == 1){
            if(flags_[0] == LDST32) width = 32;
            else if(flags_[0] == LDST64) width = 64;
            // don't need flag in codegen
            flags_.clear();
          }

          
          if(operands_.size() != 2) throw runtime_error("CS2R excepts two operands.\n");
          if(operands_[0]->state_space_ != REG || operands_[0]->getBitWidth() != width)
            throw runtime_error("CS2R's dst is invalid.\n");
          if(operands_[1]->state_space_ != SREG) throw runtime_error("CS2R's src must be sreg.\n");

          dst_operands_.emplace_back(operands_[0].release());
          src_operands_.emplace_back(operands_[1].release());
        }
        break;
      case NOP:
        {
          if(operands_.size() != 0) throw runtime_error("NOP cannot accept any operand.\n");
          if(flags_.size() != 0) throw runtime_error("NOP cannot accept any flag.\n");
        }
        break;
      case S2R:
        {
          if(flags_.size() != 0) throw runtime_error("S2R cannot accept any flag.\n");
          if(operands_.size() != 2) throw runtime_error("S2R can only accept 2 operands.\n");
          if(operands_[0]->state_space_ != REG || operands_[0]->getBitWidth() != 32)
            throw runtime_error("S2R's dst is invalid.\n");
          if(operands_[1]->state_space_ != SREG) throw runtime_error("S2R's src must be sreg.\n");

          dst_operands_.emplace_back(operands_[0].release());
          src_operands_.emplace_back(operands_[1].release());
        }
        break;
    };
  }

  //--------------------------private-functions----------------------
  void Instruction::checkFloatFlags() {
    set<Flag> const valid = {RN, RZ, RM, RP};
    if(flags_.size() > 1 || 
       (flags_.size() == 1 && valid.find(flags_[0]) == valid.end())){
      throw runtime_error("Floating point instruction has invalid flags.\n");
    }
  }

  void Instruction::checkGmemFlags(){
    // (.16|.32|.64|.128)?(.lu|.ef)?(.cta|.gpu|.sys)?
    enum FlagState {
      BEGIN, WIDTHED, CACHED, STRONGED, SCOPED,
    };

    bool has_width = false;
    bool has_cache = false;
    bool has_scope = false;
    bool has_strong = false;

    auto current_state = BEGIN;
    for(auto iter = flags_.begin(); iter != flags_.end(); ++iter){
      switch(*iter){
        case LDST16: case LDST32: case LDST64: case LDST128:
          has_width = true;
          switch(current_state){
            case BEGIN:
              current_state = WIDTHED;
              break;
            default:
              throw runtime_error("LDG/STG has invalid flags.\n");
          }
          break;

        case LU: case EF:
          has_cache = true;
          switch(current_state){
            case BEGIN: case WIDTHED:
              current_state = CACHED;
              break;
            default:
              throw runtime_error("LDG/STG has invalid flags.\n");
          }
          break;

        case MCONST: case MSTRONG: case MWEAK:
          has_strong = true;
          switch(current_state){
            case BEGIN: case WIDTHED: case CACHED:
              current_state = STRONGED;
              break;
            default:
              throw runtime_error("LDG/STG has invalid flags.\n");
          }
          break;

        case CTA: case GPU: case SYS:
          has_scope = true;
          switch(current_state){
            case BEGIN: case WIDTHED: case CACHED: case STRONGED:
              current_state = SCOPED;
              break;
            default:
              throw runtime_error("LDG/STG has invalid flags.\n");
          }
          break;
        default:
          throw runtime_error("LDG/STG has invalid flags.\n");
      }
    }

    if(!has_width) flags_.push_back(LDST32);
    if(!has_cache) flags_.push_back(CACHE_DEFAULT);
    if(!has_strong) flags_.push_back(STRONG_DEFAULT);
    if(!has_scope) flags_.push_back(SYS);

  }

  void Instruction::checkSmemFlags(){
    map<Flag, int> const ldst_width = {
      {LDST16, 16}, {LDST32, 32}, {LDST64, 64}, {LDST128, 128},
    };
    if(flags_.size() == 0 || 
      (flags_.size() == 1 && ldst_width.find(flags_[0]) != ldst_width.end())){
        // accept
    } else {
      throw runtime_error("LDS/STS has invalid flags.\n");
    }
  }

  int Instruction::getLdstWidth() const {
    set<Opcode> const ldst = {LDG, STG, LDS, STS, LDC};
    map<Flag, int> const width_flags = {
      {LDST16, 16}, {LDST32, 32}, {LDST64, 64}, {LDST128, 128}
    };

    int width = 0;

    if(ldst.find(opcode_) != ldst.end()) width = 32; // default

    for(Flag flag : flags_){
      if(width_flags.find(flag) != width_flags.end()){ 
        width = width_flags.at(flag);
        break;
      }
    }

    return width;
  }

} // namespace dada